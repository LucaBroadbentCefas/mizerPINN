from __future__ import annotations
import torch
from PINNmizer.io import load_mizer_inputs
from PINNmizer.params import active_grid_mask
from .state_scale import set_state_scale_from_initial_condition, interpolate_log_state_scale, reconstruct_scalar_state
from .residual import compute_residual_from_fields

def main():
 params,n_init,_=load_mizer_inputs('validation/fixtures/pde_multispecies', dtype=torch.float64, device='cpu')
 set_state_scale_from_initial_condition(params,n_init,eps=1e-30); old=params.state_scale_log.clone(); active=active_grid_mask(params)
 assert params.state_scale_log.shape==n_init.shape and not params.state_scale_log.requires_grad
 logn=torch.log(torch.clamp(n_init,min=1e-30)); assert torch.allclose((logn-params.state_scale_log)[active], torch.zeros_like(logn[active]))
 w=torch.sqrt(params.w[:-1]*params.w[1:])[:4]; ls,S,d=interpolate_log_state_scale(params,w); assert torch.isfinite(d).all() and torch.all(S>0)
 logU=torch.zeros(2,1,w.numel(),dtype=torch.float64); rec=reconstruct_scalar_state(logU,params,species_idx=0,w=w); assert torch.allclose(rec['N'],rec['S']*rec['U']); assert torch.allclose(rec['log_N'],rec['log_S']+rec['log_U'])
 state={k:torch.ones(2,1,w.numel(),dtype=torch.float64)*0.1 for k in ['dlogU_dt','dlogU_dw','dU_dt','dU_dw','dN_dt','dN_dw']}; state.update({'dlogS_dw':rec['dlogS_dw'],'U_eval':rec['U'],'N_eval':rec['N']})
 bio={'g':torch.ones_like(logU)*0.2,'mu_total':torch.ones_like(logU)*0.3,'dg_dw':torch.ones_like(logU)*0.4}; r=compute_residual_from_fields(state,bio); assert torch.allclose(r['residual_scaled'], state['U_eval']*r['residual_log'])
 assert torch.equal(params.state_scale_log,old); print('validate_tranche1_core passed')
if __name__=='__main__': main()
