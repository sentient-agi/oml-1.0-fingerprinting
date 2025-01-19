import random
from typing import List, Union

import accelerate
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer, move_to_device
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import FSDPModelStorage, apply_task_vector, prepare_task_vectors, obj_standard_max_next_token, next_n_batches, delete_optimizer, distributed_sample_adversary_lr, distributed_sample_task

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm
from transformers import AutoModelForCausalLM


def inner_loop_step_ft(
    model: Union[AutoModelForCausalLM, FSDP],
    adversary_batches: List[torch.Tensor],
    fingerprint_batches,
    inner_optimizer: AcceleratedOptimizer,
    inner_scheduler: torch.optim.lr_scheduler.LRScheduler,
    accelerator: Accelerator,
    gradient_accumulation_steps: int = 8,
    adv_gradient_accumulation_steps: int = 8,
    ft_grad_scale: float = 0.25,
    model_storage: FSDPModelStorage = None,
    sub_pbar: tqdm = None,
    compute_tamper_resistance_grad: bool = True,
) -> float:
    """
    Perform a single inner loop step in the tamper-resistant training process.

    This function executes an adversarial step, updates the model, and optionally computes
    the tamper resistance gradient.

    Args:
        model (Union[AutoModelForCausalLM, FSDP]): The model being trained.
        adversary_batches (List[torch.Tensor]): Batches of adversarial data.
        meta_forget_batches (List[torch.Tensor]): Batches of meta-forget data.
        dpo_pref_batches (List[torch.Tensor]): Batches of DPO preference data.
        inner_optimizer (AcceleratedOptimizer): The optimizer for the inner loop.
        inner_scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler.
        accelerator (Accelerator): The Hugging Face Accelerator object.
        gradient_accumulation_steps (int, optional): Number of steps to accumulate gradients. Defaults to 8.
        ft_grad_scale (float, optional): Scaling factor for meta gradients. Defaults to 0.25.
        model_storage (FSDPModelStorage, optional): Storage for FSDP model parameters and gradients.
        sub_pbar (tqdm, optional): Progress bar for sub-steps.
        tamper_resistance_loss_type (str, optional): Type of tamper resistance loss. Defaults to "ascent_grad".
        compute_tamper_resistance_grad (bool, optional): Whether to compute tamper resistance gradient. Defaults to True.

    Returns:
        float: The computed tamper resistance loss.
    """

    # Adversary finetuning the model
    total_adv_loss = 0
    for i in range(adv_gradient_accumulation_steps):
        output = model(**adversary_batches[i])
        loss = output.loss
        loss = loss / adv_gradient_accumulation_steps
        accelerator.backward(loss)
        total_adv_loss += loss.item()
    if accelerator.is_main_process:
        sub_pbar.update(1)
        sub_pbar.set_postfix({"adv loss": total_adv_loss})
        wandb.log({"inner_ft_loss": total_adv_loss})

    inner_optimizer.step()
    if inner_scheduler:
        inner_scheduler.step()
    model.zero_grad(set_to_none=True)

    # Compute tamper-resistance loss
    ft_loss = 0
    if compute_tamper_resistance_grad:
        total_loss = 0.0        
        for i in range(gradient_accumulation_steps):
            diagnostic_name = "next_token"
            output = model(**fingerprint_batches[i])
            loss = output.loss
            loss = loss * ft_grad_scale

            loss = loss / (gradient_accumulation_steps)
            accelerator.backward(loss)
            total_loss += loss.item()
        
        # Should this be inside the loop?
        # Keeping it inside the loop is a bug!!! Investigate why this was working!!!
        # Accumulate sharded TR loss grads in FSDPModelStorage data structure
        # This is done so everything is computed in place
        model_storage.collect_param_or_grad(
            model=model,
            accelerator=accelerator,
            to_cpu=True,  # See if we can change this to be faster?
            mode="grads",
        )
        # Clear grads from model to be ready for next adversary step
        model.zero_grad(set_to_none=False)
        ft_loss = total_loss
        if accelerator.is_main_process:
            wandb.log(
                {
                    f"inner_adv_ft_loss": total_loss / ft_grad_scale,
                    
                }
            )

    return ft_loss



def schedule(i: int = None, K: int = None, schedule_lambda: float = 0.5):
    """
    Calculate a schedule value based on the current step and total steps.

    This function computes an exponential schedule value used for weighting
    or scaling purposes during TAR.

    Args:
        i (int): The current step or iteration number.
        K (int): The total number of steps or iterations.
        schedule_lambda (float, optional): A scaling factor for the exponent. Defaults to 0.5.

    Returns:
        float: The computed schedule value as a Python float.
    """
    return torch.exp(schedule_lambda * (torch.tensor(i) - (K - 1))).item()


def _sample_switching_point(
    switching_point_coeffs: str, tar_inner_loop_steps: int
) -> int:
    coeffs = {
        k: float(v)
        for k, v in [item.split(":") for item in switching_point_coeffs.split(",")]
    }
    M = int(
        torch.distributions.Beta(coeffs["alpha"], coeffs["beta"]).sample()
        * tar_inner_loop_steps
    )
    M = accelerate.utils.broadcast_object_list([M], 0)[0]
    return M



def ft_meta_training_loop(
    model: Union[AutoModelForCausalLM, FSDP],
    dataloaders: dict[str, torch.utils.data.DataLoader],
    optimizer: AcceleratedOptimizer,
    accelerator: Accelerator,
    scheduler, #: torch.optim.lr_scheduler.LambdaLR,
    tokenizer: AutoTokenizer,
    gradient_accumulation_steps: int = 2,   
    max_steps: int = 1000,
    ft_inner_loop_steps: int = 4,
    ft_loss_scale: float = 4.0,
    inner_ft_optimizer: str = "sgd", # One of "sgd" or "adam"
    schedule_lambda: float = 0.5,
    inner_optimizer_warmup_steps: float = 20,
    use_weighting_schedule: bool = False,
    adversary_lr_schedulers: str = "constant:1.0,linear_warmup:0.25",
    adversary_lr_samples: str = "1e-5", # ,4e-5,1e-4",
    adversaries_per_step: int = 1,
    compute_adv_loss_grad_every_k_steps: int = 1,
    adv_gradient_accumulation_steps: int = 2,
    ce_loss_scale: float = 1.0,
    forgetting_regularizer_strength: float = 0.0,
    model_averaging_every_k_steps: int = 100000,
    **kwargs,
):
    
    model.config.use_cache = False
    model.train()

    # Retain and heldout forget dataloaders use `retain` and `meta` keys from dataloader funcs
    fingerprint_iterator, fingerprint_dataloader = (
        iter(dataloaders["fingerprint"]),
        dataloaders["fingerprint"],
    )

    # Adversary dataloaders use the remaining keys
    adversary_dataloaders = {
        key: {"iter": iter(value), "dataloader": value}
        for key, value in dataloaders.items()
        if key not in ["fingerprint"]
    }

    if accelerator.is_main_process:
        pbar = tqdm(
            colour="green",
            desc=f"Outer Training Loop",
            total=max_steps,
            dynamic_ncols=True,
        )

    model_storage = FSDPModelStorage()
    adversary_lr_samples = [float(lr) for lr in adversary_lr_samples.split(",")]

    # FSDP requires initial forward/backward for sharded params to be accessible in `FlatParamHandle`
    # Necessary for storing sharded params in FSDPModelStorage before the first TR step
    accelerator.backward(
        obj_standard_max_next_token(model, next(fingerprint_iterator), accelerator)
    )
    model.zero_grad(set_to_none=False)

    if forgetting_regularizer_strength > 0:
        model_storage.store_original_model(model)
    
    adv_ft_loss = 0
    for train_step in range(max_steps):
        adv_ft_loss = 0
        # Save params for tamper-resistance-optimizer step
        model_storage.collect_param_or_grad(
            model=model,
            accelerator=accelerator,
            to_cpu=True,
            mode="params",
        )
        outer_retain_batches, retain_iterator = next_n_batches(
            fingerprint_iterator, fingerprint_dataloader, gradient_accumulation_steps
        )
        optimizer.load_state_dict(move_to_device(optimizer.state_dict(), "cpu"))

        torch.cuda.empty_cache()

        
        for _ in range(adversaries_per_step):  # TODO: increase this range, make it a parameter?
            sub_pbar = None
            # adversary_type = distributed_sample_task(adversary_dist_types)
            if accelerator.is_main_process:
                sub_pbar = tqdm(
                    colour="blue",
                    desc=f"Step - {train_step} : Inner Training Loop ",
                    total=ft_inner_loop_steps,
                    dynamic_ncols=True,
                )
            adversary_lr_scheduler = distributed_sample_task(adversary_lr_schedulers)


            # Sample adversary learning rate
            adversary_lr = distributed_sample_adversary_lr(
                adversary_lr_samples, accelerator
            )

            # Setup adversary optimizer
            if inner_ft_optimizer == "adam":
                inner_optimizer = torch.optim.AdamW(model.parameters(), lr=adversary_lr)
            elif inner_ft_optimizer == "sgd":
                inner_optimizer = torch.optim.SGD(model.parameters(), lr=adversary_lr)
            else:
                raise ValueError("Invalid inner optimizer; must be 'adam' or 'sgd'")
            inner_optimizer = accelerator.prepare_optimizer(inner_optimizer)
            inner_scheduler = None

            # Setup adversary learning rate scheduler
            if adversary_lr_scheduler == "linear_warmup":
                inner_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    inner_optimizer,
                    lambda step: ft_inner_loop_steps / inner_optimizer_warmup_steps,
                )
                inner_scheduler = accelerator.prepare(inner_scheduler)

            for inner_step in range(ft_inner_loop_steps):
                # Sample adversary batches
                _adversary_type = "alpaca"  # adversary_type
                (
                    adversary_batches,
                    adversary_dataloaders[_adversary_type]["iter"],
                ) = next_n_batches(
                    adversary_dataloaders[_adversary_type]["iter"],
                    adversary_dataloaders[_adversary_type]["dataloader"],
                    adv_gradient_accumulation_steps,
                )

                # Per-step tamper-resistance loss weighting schedule
                scheduled_weighting = (
                    schedule(inner_step, ft_inner_loop_steps, schedule_lambda)
                    if use_weighting_schedule
                    else 1 / ft_inner_loop_steps
                )

                # Whether to compute TR grad for current step (sub-sampling trick from appendix)
                compute_tamper_resistance_grad = (
                    inner_step + 1
                ) % compute_adv_loss_grad_every_k_steps == 0

                # Compute adversary step and tamper-resistance loss
                adv_ft_loss += inner_loop_step_ft(
                    model=model,
                    adversary_batches=adversary_batches,
                    fingerprint_batches=outer_retain_batches,
                    inner_optimizer=inner_optimizer,
                    inner_scheduler=inner_scheduler,
                    accelerator=accelerator,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    adv_gradient_accumulation_steps=adv_gradient_accumulation_steps,
                    ft_grad_scale=ft_loss_scale
                    * scheduled_weighting
                    / 1,
                    model_storage=model_storage,
                    sub_pbar=sub_pbar,
                    compute_tamper_resistance_grad=compute_tamper_resistance_grad,

                )

            # accelerator.print(f"Completed Inner loop")
            # inner_optimizer.load_state_dict(
            #     move_to_device(inner_optimizer.state_dict(), "cpu")
            # )
            delete_optimizer(inner_optimizer)

            model_storage.add_from_storage_to_model(
                model=model,
                accelerator=accelerator,
                skip_check=True,
                mode="params",
            )
            # model_storage.offload_params_or_grads("grads")
            torch.cuda.empty_cache()
            
        optimizer.load_state_dict(
            move_to_device(optimizer.state_dict(), optimizer.accelerator_state.device)
        )


        total_retain_loss = 0
        for i in range(gradient_accumulation_steps):
            outputs = model(**outer_retain_batches[i])
            retain_loss = outputs.loss
            retain_loss = retain_loss / gradient_accumulation_steps * ce_loss_scale
            accelerator.backward(retain_loss)
            total_retain_loss += retain_loss.item()
            
        # Add tamper-resistance gradients to model
        # if (
        #     tamper_resistance_loss >= tar_tamper_resistance_loss_lower_bound
        #     or unbounded
        # ):
        model_storage.add_from_storage_to_model(
            model=model,
            accelerator=accelerator,
            mode="grads",
            skip_check=ft_inner_loop_steps==0,  
        )

        # Clear from storage to reduce peak memory usage
        model_storage.clear_grads()
        model_storage.clear_params()

        # Tamper-resistance meta-optimizer step
        optimizer.step()
        scheduler.step()
        model.zero_grad(set_to_none=True)
        
        if forgetting_regularizer_strength > 0 and train_step % model_averaging_every_k_steps == 0 and train_step > 0:
            model_storage.merge_original_model(model, forgetting_regularizer_strength)
        
        if accelerator.is_main_process:
            pbar.update(1)
            pbar.set_postfix(
                {
                    "fingerprinting loss / adv_ft_loss": f"{total_retain_loss} / {adv_ft_loss}"
                }
            )
            wandb.log(
                {
                    "step": train_step,
                    "fingerprint_loss": total_retain_loss / ce_loss_scale,
                    "adv_ft_loss": adv_ft_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            # if train_step % 10 == 0:
            #     eval_pbar = tqdm(
            #         colour="red",
            #         desc=f"Eval Loop",
            #         total=512,
            #         dynamic_ncols=True,)
            #     fingerprint_acc, frac_acc = eval_backdoor_acc(model, tokenizer, dataloaders["fingerprint_ds_for_eval"], prompt_templates=["{}"], pbar=eval_pbar)
            #     wandb.log({"fingerprint_acc": fingerprint_acc, "frac_acc": frac_acc})
        # accelerator.print(f"Completed Outer loop")
    print("Training completed")
    return model

def inner_step_task_vectors(
    model: Union[AutoModelForCausalLM, FSDP],
    task_vectors,
    task_vector_coefficients,
    fingerprint_batches,
    accelerator: Accelerator,
    gradient_accumulation_steps: int = 8,
    ft_grad_scale: float = 0.25,
    model_storage: FSDPModelStorage = None,
):  
    total_loss = 0.0
    for tv_idx in range(len(task_vector_coefficients)):
        model_storage.add_from_storage_to_model(
            model=model,
            accelerator=accelerator,
            skip_check=True,
            mode="params",
        )
        task_vector_coefficient = task_vector_coefficients[tv_idx]
        
        model = apply_task_vector(model, task_vectors, task_vector_coefficient)
        for i in range(gradient_accumulation_steps):
            output = model(**fingerprint_batches[i])
            loss = output.loss
            loss = loss * ft_grad_scale

            loss = loss / (gradient_accumulation_steps * len(task_vector_coefficients))
            accelerator.backward(loss)
            total_loss += loss.item()

        model_storage.collect_param_or_grad(
            model=model,
            accelerator=accelerator,
            to_cpu=True,  # See if we can change this to be faster?
            mode="grads",
        )
        model.zero_grad(set_to_none=False)
    ft_loss = total_loss
        # if accelerator.is_main_process:
        #     wandb.log(
        #         {
        #             f"inner_adv_ft_loss": total_loss / ft_grad_scale,
                    
        #         }
        #     )
        
    return ft_loss


def task_vectors_training_loop(
    model: Union[AutoModelForCausalLM, FSDP],
    dataloaders: dict[str, torch.utils.data.DataLoader],
    optimizer: AcceleratedOptimizer,
    accelerator: Accelerator,
    scheduler, #: torch.optim.lr_scheduler.LambdaLR,
    tokenizer: AutoTokenizer,
    model_tv: Union[AutoModelForCausalLM, FSDP], # Instruction-Tuned model (for task vectors)
    task_vectors_coefficients, # Can be a list  
    gradient_accumulation_steps: int = 2,   
    max_steps: int = 1000,
    ft_loss_scale: float = 4.0,
    ce_loss_scale: float = 1.0,

    **kwargs,
    
)  :
    model.config.use_cache = False
    model.train()

    # Retain and heldout forget dataloaders use `retain` and `meta` keys from dataloader funcs
    fingerprint_iterator, fingerprint_dataloader = (
        iter(dataloaders["fingerprint"]),
        dataloaders["fingerprint"],
    )


    if accelerator.is_main_process:
        pbar = tqdm(
            colour="green",
            desc=f"Outer Training Loop",
            total=max_steps,
            dynamic_ncols=True,
        )

    model_storage = FSDPModelStorage()

    # FSDP requires initial forward/backward for sharded params to be accessible in `FlatParamHandle`
    # Necessary for storing sharded params in FSDPModelStorage before the first TR step
    accelerator.backward(
        obj_standard_max_next_token(model, next(fingerprint_iterator), accelerator)
    )
    accelerator.backward(
        obj_standard_max_next_token(model_tv, next(fingerprint_iterator), accelerator)
    )
    model_tv.zero_grad(set_to_none=False)
    model.zero_grad(set_to_none=False)
    accelerator.print("Preparing task vectors")

    task_vectors = prepare_task_vectors(model, model_tv, model_storage)
    
    if isinstance(task_vectors_coefficients, str):
        task_vectors_coefficients = [float(tv) for tv in task_vectors_coefficients.split(",")]
    
    
    adv_ft_loss = 0
    for train_step in range(max_steps):
        model_storage.collect_param_or_grad(
            model=model,
            accelerator=accelerator,
            to_cpu=True,
            mode="params",
        )
        outer_retain_batches, retain_iterator = next_n_batches(
            fingerprint_iterator, fingerprint_dataloader, gradient_accumulation_steps
        )
        optimizer.load_state_dict(move_to_device(optimizer.state_dict(), "cpu"))

        torch.cuda.empty_cache()

        # Apply task vectors
        
        # Compute loss and gradient
        
        # Save gradients to model storage
        
        adv_ft_loss = inner_step_task_vectors(model=model,
            task_vectors=task_vectors,
            task_vector_coefficients=task_vectors_coefficients,
            fingerprint_batches=outer_retain_batches,
            accelerator=accelerator,
            gradient_accumulation_steps=gradient_accumulation_steps,
            ft_grad_scale=ft_loss_scale,
            model_storage=model_storage,
        )
        
        optimizer.load_state_dict(
            move_to_device(optimizer.state_dict(), optimizer.accelerator_state.device)
        )
        model_storage.add_from_storage_to_model(
            model=model,
            accelerator=accelerator,
            skip_check=True,
            mode="params",
        )

        total_retain_loss = 0
        for i in range(gradient_accumulation_steps):
            outputs = model(**outer_retain_batches[i])
            retain_loss = outputs.loss
            print(retain_loss)
            retain_loss = retain_loss / gradient_accumulation_steps * ce_loss_scale
            accelerator.backward(retain_loss)
            total_retain_loss += retain_loss.item()

        model_storage.add_from_storage_to_model(
            model=model,
            accelerator=accelerator,
            mode="grads",
            skip_check=False,  
        )

        # Clear from storage to reduce peak memory usage
        model_storage.clear_grads()
        model_storage.clear_params()

        # Tamper-resistance meta-optimizer step
        optimizer.step()
        scheduler.step()
        model.zero_grad(set_to_none=True)
        if accelerator.is_main_process:
            pbar.update(1)
            pbar.set_postfix(
                {
                    "fingerprinting loss / task_vectors_loss": f"{total_retain_loss} / {adv_ft_loss}"
                }
            )
            wandb.log(
                {
                    "step": train_step,
                    "fingerprint_loss": total_retain_loss / ce_loss_scale,
                    "task_vectors_loss": adv_ft_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )


    return model
