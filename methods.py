import click
import ast
import kcap_methods as km
import score_compression as sc

@click.group()
def cli():
    pass

@click.command()
@click.option('--mock_run', default=0, help='The mock run number to run derivatives on')
@click.option('--param_to_vary', prompt='Param To Vary', help='The parameter to vary')
@click.option('--params_to_fix', multiple=True, required=True, help='The parameters to keep fixed')
@click.option('--vals_to_diff', multiple=True, required=True, help='The values to differentiate')
@click.option('--step_size', prompt='Step Size', help='The step size to take stencils over')
@click.option('--stencil_pts', prompt='Number of Stencil Points', help='The number of stencil points')
@click.option('--mocks_dir', prompt='Mocks Directory', help='Restults directory')
@click.option('--mocks_name', prompt='Mocks Name', help='Root name of the KCAP mock run')
@click.option('--cleanup', default = 2, help='The cleanup option')
def run_kcap_deriv(mock_run, param_to_vary, params_to_fix, vals_to_diff, step_size, stencil_pts, mocks_dir, mocks_name, cleanup):
    """Runs KCAP derivatives for the specified KCAP simulation"""
    km.run_kcap_deriv(mock_run = mock_run, param_to_vary = param_to_vary, 
                      params_to_fix = params_to_fix, vals_to_diff = vals_to_diff, 
                      step_size = step_size, stencil_pts = stencil_pts, 
                      mocks_dir = mocks_dir, mocks_name = mocks_name, cleanup = cleanup)

cli.add_command(run_kcap_deriv)

@click.command()
@click.option('--mock_run', default=0, help='The mock run number to run derivatives on')
@click.option('--params_varied', prompt='Params Varied', help='The parameters varied')
@click.option('--vals_to_diff', multiple=True, required=True, help='The values to differentiate')
@click.option('--mocks_dir', prompt='Mocks Directory', help='Restults directory')
@click.option('--mocks_name', prompt='Mocks Name', help='Root name of the KCAP mock run')
def run_omega_m_deriv(mock_run, params_varied, vals_to_diff,  mocks_dir, mocks_name):
    """Runs KCAP omega_m derivatives for the specified KCAP simulation"""
    km.run_omega_m_deriv(mock_run = mock_run, params_varied = param_to_vary, 
                      vals_to_diff = vals_to_diff, mocks_dir = mocks_dir, mocks_name = mocks_name)

cli.add_command(run_omega_m_deriv)

@click.command()
@click.option('--mock_run', default=0, help='The mock run number to run derivatives on')
@click.option('--vals_to_read', prompt='Values to Read', help='The values to read')
@click.option('--mocks_dir', prompt='Mocks Directory', help='Restults directory')
@click.option('--mocks_name', prompt='Mocks Name', help='Root name of the KCAP mock run')
@click.option('--bin_order', default=None, help='The order of bins to read values from')
def get_values(mock_run, vals_to_read, mocks_dir, mocks_name, bin_order):
    """Gets particular values from a KCAP run"""
    values = km.get_values(mock_run = mock_run, vals_to_read = vals_to_read, mocks_dir = mocks_dir, mocks_name = mocks_name, bin_order = bin_order)
    click.echo(f"{values}")
    return values

cli.add_command(get_values)

@click.command()
@click.option('--mock_run_start', default=0, help='The mock run number to run derivatives on')
@click.option('--num_mock_runs', type=int, prompt='Values to Read', help='The values to read')
@click.option('--mocks_dir', prompt='Mocks Directory', help='Results directory')
@click.option('--mocks_name', prompt='Mocks Name', help='Root name of the KCAP mock run')
@click.option('--folders_to_keep', '-ftk', multiple=True, default=["shear_xi_minus_binned", \
                                                                   "shear_xi_plus_binned", \
                                                                   "cosmological_parameters", \
                                                                   "intrinsic_alignment_parameters", \
                                                                   "growth_parameters","bias_parameters", \
                                                                   "halo_model_parameters", \
                                                                   "likelihoods", \
                                                                   "theory_data_covariance"], 
                                                                   help='The order of bins to read values from')
@click.option('--files_to_remove', '-ftr', multiple=True, default=["theory_data_covariance/covariance.txt", "theory_data_covariance/inv_covariance.txt"], help='Extra specific files to remove')
def cleanup_folders(mock_run_start, num_mock_runs, mocks_dir, mocks_name, folders_to_keep, files_to_remove):
    """Does folder cleanup after some KCAP runs"""
    values = km.cleanup_folders(mock_run_start = mock_run_start, num_mock_runs = num_mock_runs, 
                                mocks_dir = mocks_dir, mocks_name = mocks_name,
                                folders_to_keep = folders_to_keep,
                                files_to_remove = files_to_remove)

cli.add_command(cleanup_folders)

if __name__ == '__main__':
    cli()