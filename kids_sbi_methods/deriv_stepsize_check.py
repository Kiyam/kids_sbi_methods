from . import kcap_methods as km
from . import score_compression as sc
import numpy as np


def run_varying_stepsize(
    step_sizes,
    stencil_pts=5,
    param_to_vary="cosmological_parameters--sigma_8_input",
    params_to_fix=[
        "cosmological_parameters--omch2",
        "intrinsic_alignment_parameters--a",
        "cosmological_parameters--n_s",
        "halo_model_parameters--a",
        "cosmological_parameters--h0",
        "cosmological_parameters--ombh2",
        "nofz_shifts--bias_1",
        "nofz_shifts--bias_2",
        "nofz_shifts--bias_3",
        "nofz_shifts--bias_4",
        "nofz_shifts--bias_5",
    ],
    vals_to_diff=["bandpowers--bandpowers", "bandpowers_novd--bandpowers"],
    mocks_dir="/share/lustre/klin/kcap_out/deriv_tests",
    mocks_root_name="glass_mocks",
    deriv_dir="/share/lustre/klin/kcap_out/kids_1000_mock_derivatives",
    deriv_name="glass_salmo_deriv_sims",
    deriv_ini_file="/share/lustre/klin/kids_sbi/runs/sbi_config/glass_salmo_deriv_pipeline_pcl.ini",
    deriv_values_file="/share/lustre/klin/kids_sbi/runs/sbi_config/kids_glass_deriv_values.ini",
    deriv_values_list_file="/share/lustre/klin/kids_sbi/runs/sbi_config/kids_glass_deriv_values_list.ini",
    sbatch_file="/share/rcifdata/klin/sbi_sim_deriv.sh",
):
    for step_size in step_sizes:
        km.run_kcap_deriv(
            param_to_vary=param_to_vary,
            params_to_fix=params_to_fix,
            vals_to_diff=vals_to_diff,
            step_size=float(step_size),
            stencil_pts=stencil_pts,
            mocks_dir=mocks_dir,
            mocks_name=mocks_root_name + "_" + str(step_size),
            cleanup=True,
            deriv_dir=deriv_dir,
            deriv_name=deriv_name,
            deriv_ini_file=deriv_ini_file,
            deriv_values_file=deriv_values_file,
            deriv_values_list_file=deriv_values_list_file,
            sbatch_file=sbatch_file,
        )


def get_fom_vals(
    step_size_names, inv_covariance, deriv_params, data_params, mocks_dir, mocks_name
):
    """
    Calculates the fom values
    """

    inv_covariance = np.loadtxt(inv_covariance)
    fom = np.zeros((len(step_size_names), len(deriv_params)))

    for i, step_size in enumerate(step_size_names):
        deriv_matrix = km.get_fiducial_deriv(
            deriv_params=deriv_params,
            data_params=data_params,
            fiducial_run=step_size,
            mocks_dir=mocks_dir,
            mocks_name=mocks_name,
        )
        fisher_matrix = sc.calc_fisher(
            inv_covariance=inv_covariance, deriv_matrix=deriv_matrix
        )
        fom[i] += np.sqrt(1 / np.diag(fisher_matrix))

    return fom
