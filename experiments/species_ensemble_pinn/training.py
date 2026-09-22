from __future__ import annotations

import torch

from .losses import compute_composite_loss
from .pde_state import compute_pde_state
from .residual import compute_pde_residual_from_state
from .sampling import sample_pde_batch
from .weighting import update_expert_gradient_norm_weights_


def _loss_bundle(model, batch, context, loss_weights):
    state = compute_pde_state(model, batch, context.params, context.n_init, context.n_pp,
                              context.known_state, species_idx=context.species_idx)
    residual = compute_pde_residual_from_state(state, context.params, species_idx=context.species_idx)
    return compute_composite_loss(state=state,residual_out=residual,params=context.params,
        species_idx=context.species_idx,batch=batch,lambda_pde=context.args.lambda_pde,
        lambda_ic=context.args.lambda_ic,lambda_bc=context.args.lambda_bc,loss_weights=loss_weights,
        causal_n_chunks=context.args.causal_n_chunks,causal_epsilon=context.args.causal_epsilon,
        eps=context.args.loss_eps,bc_g_min=context.args.bc_g_min)


def train_one_step(*, model, optimizer, scheduler, context, step: int,
                   t_max_current: float, loss_weights: dict[str,float], fixed_weight_batch: dict) -> dict:
    optimizer.zero_grad(set_to_none=True)
    batch=sample_pde_batch(context.params,context.args.n_time,context.args.n_eval,
        t_max_current=t_max_current,causal_n_chunks=context.args.causal_n_chunks)
    loss, out=_loss_bundle(model,batch,context,loss_weights)
    weight_stats={}
    if step % context.args.expert_weight_update_every == 0:
        _, fixed=_loss_bundle(model,fixed_weight_batch,context,loss_weights)
        weight_stats=update_expert_gradient_norm_weights_(model,
            {"pde":context.args.lambda_pde*fixed["loss_pde"],"ic":context.args.lambda_ic*fixed["loss_ic"],"bc":context.args.lambda_bc*fixed["loss_bc"]},
            loss_weights,alpha=context.args.expert_weight_alpha,min_weight=context.args.weight_min,max_weight=context.args.weight_max)
        loss,out=_loss_bundle(model,batch,context,loss_weights)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite training loss.")
    loss.backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Non-finite or missing model gradients.")
    grad_norm=float(torch.sqrt(sum(torch.sum(g.detach()**2) for g in grads)).cpu())
    optimizer.step()
    scheduler.step()
    return {"step":step,"lr":optimizer.param_groups[0]["lr"],"loss":float(loss.detach().cpu()),
        "loss_unweighted":float(out["loss_unweighted"].detach().cpu()),"loss_pde":float(out["loss_pde"].detach().cpu()),
        "loss_ic":float(out["loss_ic"].detach().cpu()),"loss_bc":float(out["loss_bc"].detach().cpu()),
        "loss_pde_ungated":float(out["loss_pde_ungated"].detach().cpu()),"w_pde":loss_weights["pde"],
        "w_ic":loss_weights["ic"],"w_bc":loss_weights["bc"],"grad_norm":grad_norm,**weight_stats}
