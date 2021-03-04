# setup of KCAP environment
# +++ run from within bash shell! +++
# +++ call with 'source ./kcap_setup.sh' +++

echo "Setting up KCAP environment (make sure this is run from within bash and prepended by 'source'!) ..."


# source /etc/profile.d/modules.sh   # load so module command is known   
eval "$(conda shell.bash hook)"  #prepares anaconda                                                           
# module purge
unset COSMOSIS_SRC_DIR
unset PYTHONPATH
# module load dev_tools/sep2019/cmake-3.15.3 dev_tools/sep2019/openmpi-4.0.1

export ENVLOC=/home/ruyi_wsl/anaconda3/envs/kcap_env
export COSMOSIS_SRC_DIR=/home/ruyi_wsl/kcap
export KIDS_MOCKS_DIR=/home/ruyi_wsl/kcap/kids_1000_mocks
export KIDS_DERIV_DIR=/home/ruyi_wsl/kcap/kids_1000_mock_derivatives
export KIDS_MOCKS_ROOT_NAME=kids_1000_cosmology
export KIDS_DERIV_ROOT_NAME=kids_derivs
export KIDS_PIPELINE_INI_FILE=/home/ruyi_wsl/kcap/runs/lfi_config/kids_pipeline_grid.ini
export KIDS_PIPELINE_VALUES_FILE=/home/ruyi_wsl/kcap/runs/lfi_config/kids_values_grid.ini
export KIDS_DERIV_INI_FILE=/home/ruyi_wsl/kcap/runs/lfi_config/kids_deriv_pipeline.ini
export KIDS_DERIV_VALUES_FILE=/home/ruyi_wsl/kcap/runs/lfi_config/kids_deriv_values.ini
export KIDS_PIPELINE_DERIV_VALUES_LIST=/home/ruyi_wsl/kcap/runs/lfi_config/kids_deriv_values_list.ini
export GSL_INCLUDE_DIRS=$ENVLOC/include
export GSL_LIB_DIRS=$ENVLOC/lib
export GSL_ROOT_DIR=$ENVLOC
conda activate kcap_env

echo "Done."
