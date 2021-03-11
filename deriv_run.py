import numpy as np
import scipy as sc
import configparser as cfg
import subprocess
import sys
import os
import glob
import tarfile
import shutil
import errno

#TODO - Need to set it so that the stepsize is a relative stepsize, typically of order 1*10^-2 -> 1*10^-5

class kcap_deriv:
    def __init__(self, mock_run, param_to_vary, params_to_fix, vals_to_diff):
        """
        Gets variable from shell
        """
        self.cosmosis_src_dir = os.environ['COSMOSIS_SRC_DIR']
        self.kids_mocks_dir = os.environ['KIDS_MOCKS_DIR']
        self.kids_deriv_dir = os.environ['KIDS_DERIV_DIR']
        self.kids_mocks_root_name = os.environ['KIDS_MOCKS_ROOT_NAME']
        self.kids_deriv_root_name = os.environ['KIDS_DERIV_ROOT_NAME']
        self.kids_pipeline_ini_file = os.environ['KIDS_PIPELINE_INI_FILE']
        self.kids_pipeline_values_file = os.environ['KIDS_PIPELINE_VALUES_FILE']
        self.kids_deriv_ini_file = os.environ['KIDS_DERIV_INI_FILE']
        self.kids_deriv_values_file = os.environ['KIDS_DERIV_VALUES_FILE']
        self.kids_deriv_values_list = os.environ['KIDS_PIPELINE_DERIV_VALUES_LIST']
        self.mock_run = self.check_mock_run_exists(mock_run)
        
        if param_to_vary in params_to_fix:
            raise Exception("Specified parameter to vary is also specified to not vary, inconsistent settings")
        
        if isinstance(params_to_fix, list):
            self.params_to_fix = params_to_fix
        else:
            raise Exception("params_to_fix variable must be a list")
        
        if isinstance(param_to_vary, str):
            self.param_to_vary = param_to_vary
        else:
            raise Exception("param_to_vary variable must be a string")

        if isinstance(vals_to_diff, list):
            self.vals_to_diff = vals_to_diff
        else:
            raise Exception("vals_to_diff variable must be a list")

        self.parameter_list = params_to_fix + [param_to_vary]

        self.param_header, self.param_name = param_to_vary.split("--")

    def check_mock_run_exists(self, mock_run):
        """
        Checks if the requested mock_run file exists, and if not will check for a .tgz file of the same file name and untar as necessary
        """
        if type(mock_run) is int:
            mock_run = str(mock_run)
        elif type(mock_run) is str:
            pass
        else:
            raise Exception("Sorry, the requested mock run is named incorrectly")

        if os.path.exists(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+mock_run):
            pass
        elif os.path.exists(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+mock_run+'.tgz'):
            print("Need to untar files first...")
            self.extract_tar(mock_run)
            print("Succesfully untarred")
        else:
            raise Exception("Sorry, the requested mock run doesn't exist")
        return mock_run

    def extract_tar(self, mock_run, option = "mocks"):
        """
        Untars file
        """
        if option == "mocks":
            kcap_tar = tarfile.open(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+mock_run+'.tgz')
            kcap_tar.extractall(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+mock_run)
            kcap_tar.close()
        elif option == "derivs":
            for step in range(4):
                kcap_tar = tarfile.open(self.kids_deriv_dir+'/'+self.kids_deriv_root_name+'_'+str(step)+'.tgz')
                kcap_tar.extractall(self.kids_deriv_dir+'/'+self.kids_deriv_root_name+'_'+str(step))
                kcap_tar.close()
        else:
            raise Exception("Ill defined option passed as argument to extract_tar method")      

    def get_params(self):
        """
        Gets parameters from the specified mock run
        """
        parameter_dict = {} # Dictionary of parameters to write to file
        for item in self.parameter_list:
            header, name = item.split("--")
            param_val = self.read_param_from_txt_file(file_location = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+header+'/values.txt', parameter = name)
            parameter_dict[item] = param_val
        print("Fetched parameters are: %s" % str(parameter_dict))
        self.param_dict = parameter_dict
        return(parameter_dict)

    def read_param_from_txt_file(self, file_location, parameter):
        """
        Given a specific txt file, will read the parameter from it as defined by input parameter
        """
        parameters_file = open(file_location)
        for line in parameters_file:
            key, value = [word.strip() for word in line.split("=")]
            if key == parameter:
                param_val = float(value)
        parameters_file.close()
        return(param_val)

    def write_deriv_values(self, step_size):
        """
        Modifies the deriv_values_list.ini file
        """
        values_config = cfg.ConfigParser()
        values_config.read(self.kids_pipeline_values_file)

        for param in self.params_to_fix:
            if param in self.param_dict:
                header, name = param.split("--")
                if name == "sigma_8":
                    name = "sigma_8_input"
                values_config[header][name] = str(self.param_dict[param])
            else:
                raise Exception("Unknown parameter specified in params_to_fix")

        if self.param_to_vary in self.param_dict:
            middle_val = self.param_dict[self.param_to_vary]
            abs_step_size = middle_val * step_size
            lower_two_step = middle_val - 2*abs_step_size
            lower_one_step = middle_val - abs_step_size
            up_one_step = middle_val + abs_step_size
            up_two_step = middle_val + 2*abs_step_size
            
            values_list_file = open(self.kids_deriv_values_list, "w")
            file_text = ["#"+self.param_to_vary+"\n", str(lower_two_step)+"\n"+str(lower_one_step)+"\n"+str(up_one_step)+"\n"+str(up_two_step)]
            values_list_file.writelines(file_text)
            values_list_file.close()

            new_param_string = str(lower_two_step) + " " + str(middle_val) + " " + str(up_two_step)

            if self.param_name == "sigma_8":
                name = "sigma_8_input"
            else:
                name = self.param_name
            values_config[self.param_header][name] = new_param_string
        else:
            raise Exception("Badly defined parameter to vary...")

        with open(self.kids_deriv_values_file, 'w') as configfile:
            values_config.write(configfile)
        
        return step_size, abs_step_size

    def run_deriv_kcap(self, mpi_opt, threads):
        if mpi_opt == True:
            if isinstance(threads, int):
                subprocess.run(["mpirun", "-n" , str(threads), "cosmosis", "--mpi", self.kids_deriv_ini_file])
            else:
                raise Exception("Incorrect number of threads requested for MPI")
        elif mpi_opt == False:
            subprocess.run(["cosmosis", self.kids_deriv_ini_file])
        else: 
            raise Exception

    def copy_deriv_vals_to_mocks(self, step_size, abs_step_size):
        if len(glob.glob(self.kids_deriv_dir+'/'+self.kids_deriv_root_name+'_*/')) == 4:
            pass
        elif len(glob.glob(self.kids_deriv_dir+'/'+self.kids_deriv_root_name+'_*.tgz')) == 4:
            print("Need to untar files first...")
            self.extract_tar(self.mock_run, option = "derivs")
            print("Succesfully untarred")
        else:
            raise Exception("Sorry, you seem to be missing the derivatives, try running KCAP for the requested derivative steps first")
        
        step_list = ["-2dx", "-1dx", "+1dx", "+2dx"]

        stepsize_file = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/deriv_stepsizes/'+self.param_name+"_stepsize.txt"
        if not os.path.exists(os.path.dirname(stepsize_file)):
            try:
                os.makedirs(os.path.dirname(stepsize_file))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        with open(stepsize_file, "w") as f:
            f.write(self.param_to_vary+"_relative_stepsize="+str(step_size) + "\n" + self.param_to_Vary+"_absolute_stepsize="+str(abs_step_size))

        for deriv_run in range(4):
            for param in self.vals_to_diff:
                new_subdir_root = param + "_" + self.param_name
                shutil.copytree(self.kids_deriv_dir+'/'+self.kids_deriv_root_name+'_'+str(deriv_run)+'/'+param, 
                                self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+new_subdir_root+step_list[deriv_run])
    
    def check_existing_derivs(self):
        print("Checking if the corresponding derivatives exist...")
        check_count = 0
        for deriv_vals in self.vals_to_diff:
            num_bins = len(glob.glob(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'/*.txt')) - 2
            num_found_bins = len(glob.glob(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_name+'_deriv/*.txt'))
            if num_found_bins == num_bins:
                print("Files for %s numerical derivative values wrt to %s found." % (deriv_vals, self.param_name))
                check_count += 1
            else:
                print("Missing derivatives for %s wrt to %s." % (deriv_vals, self.param_name))
        
        stepsize_file = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/deriv_stepsizes/'+self.param_name+"_stepsize.txt"
        if os.path.exists(os.path.dirname(stepsize_file)):
            print("Stepsize file for deriv wrt to %s found" % self.param_name)
            check_count += 1
        else:
            print("Missing stepsize file for %s derivatives" % self.param_name)
        
        if check_count == len(self.vals_to_diff) + 1:
            print("All wanted numerical derivative values found!")
            return True
        else:
            return False

    def cleanup(self):
        print("Checking all files exist as expected and cleaning up...")
        files_found = self.check_existing_derivs()
        if files_found is True:
            print("Initiating cleanup...")
            shutil.rmtree(self.kids_deriv_dir)
            os.makedirs(self.kids_deriv_dir)
            check_dir = os.listdir(self.kids_deriv_dir)
            for deriv_vals in self.vals_to_diff:
                shutil.rmtree(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_name+'-2dx/')
                shutil.rmtree(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_name+'-1dx/')
                shutil.rmtree(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_name+'+2dx/')
                shutil.rmtree(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_name+'+1dx/')
            
            if len(check_dir) == 0:
                if len(glob.glob(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/*_'+self.param_name+'*dx')) == 0:
                    print("Derivatives temprorary files succesfully cleaned up.")
            else:
                raise Exception("Error during directory cleanup, please manually inspect!")
        else:
            print("Not all files found, exiting cleanup. Please manually inspect!")
            exit()

    def first_deriv(self, abs_step_size):
        """
        Calculates the first derivative using a 5 point stencil
        """
        for deriv_vals in self.vals_to_diff:
            minus_2dx_files = glob.glob(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_name+'-2dx/bin*.txt')
            minus_1dx_files = glob.glob(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_name+'-1dx/bin*.txt')
            plus_2dx_files = glob.glob(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_name+'+2dx/bin*.txt')
            plus_1dx_files = glob.glob(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_name+'+1dx/bin*.txt')

            assert len(minus_2dx_files) == len(minus_1dx_files) == len(plus_2dx_files) == len(plus_1dx_files), "Some dx stepsize files missing."
            
            #fetch bin names
            bin_names = [bin_name.split("/")[-1].replace(".txt", "") for bin_name in minus_2dx_files]

            minus_2dx_vals = np.array([])
            for dx_file_name in minus_2dx_files:
                dx_file = open(dx_file_name, "r")
                values = np.array(dx_file.read().split('\n')[1:-1]) # the [1:-1] is to remove the first line that defines which bin and the -1 is to remove trailing empty line
                minus_2dx_vals = np.append(minus_2dx_vals, values)
            minus_2dx_vals = minus_2dx_vals.astype(np.float)
            minus_2dx_vals = minus_2dx_vals.reshape((len(bin_names), -1))
            
            minus_1dx_vals = np.array([])
            for dx_file_name in minus_1dx_files:
                dx_file = open(dx_file_name, "r")
                values = np.array(dx_file.read().split('\n')[1:-1])
                minus_1dx_vals = np.append(minus_1dx_vals, values)
            minus_1dx_vals = minus_1dx_vals.astype(np.float)
            minus_1dx_vals = minus_1dx_vals.reshape((len(bin_names), -1))
            
            plus_2dx_vals = np.array([])
            for dx_file_name in plus_2dx_files:
                dx_file = open(dx_file_name, "r")
                values = np.array(dx_file.read().split('\n')[1:-1])
                plus_2dx_vals = np.append(plus_2dx_vals, values)
            plus_2dx_vals = plus_2dx_vals.astype(np.float)
            plus_2dx_vals = plus_2dx_vals.reshape((len(bin_names), -1))
            
            plus_1dx_vals = np.array([])
            for dx_file_name in plus_1dx_files:
                dx_file = open(dx_file_name, "r")
                values = np.array(dx_file.read().split('\n')[1:-1])
                plus_1dx_vals = np.append(plus_1dx_vals, values)
            plus_1dx_vals = plus_1dx_vals.astype(np.float)
            plus_1dx_vals = plus_1dx_vals.reshape((len(bin_names), -1))

            print("All values needed for %s derivatives wrt to %s found, calculating and saving to file..." % (deriv_vals, self.param_name))

            first_deriv_vals = (1/12*minus_2dx_vals - 2/3*minus_1dx_vals + 2/3*plus_1dx_vals -1/12*plus_2dx_vals)/abs_step_size

            deriv_dir_path = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_name+"_deriv/"
            if not os.path.exists(os.path.dirname(deriv_dir_path)):
                try:
                    os.makedirs(os.path.dirname(deriv_dir_path))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            for i, bin_vals in enumerate(first_deriv_vals):
                deriv_file = deriv_dir_path+bin_names[i]+".txt"
                np.savetxt(deriv_file, bin_vals, newline="\n", header=bin_names[i])
        print("Derivatives saved succesfully")

class read_kcap(kcap_deriv):
    def __init__(self, mock_run, vals_to_read):
        self.kids_mocks_dir = os.environ['KIDS_MOCKS_DIR']
        self.kids_mocks_root_name = os.environ['KIDS_MOCKS_ROOT_NAME']
        self.mock_run = self.check_mock_run_exists(mock_run)
        if isinstance(vals_to_read, list):
            self.vals_to_read = vals_to_read
        elif isinstance(vals_to_read, str):
            self.vals_to_read = [vals_to_read]
        else:
            raise Exception("Badly defined values to read, needs to be of type string or list")

    def read_vals(self):
        vals_dict = {}
        for vals in self.vals_to_read:
            files_list = glob.glob(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+vals+'/bin*.txt')
            vals_array = np.array([])
            for file_name in files_list:
                temp_file_read = open(file_name, "r")
                values = np.array(temp_file_read.read().split('\n')[1:-1]) # the [1:-1] is to remove the first line that defines which bin and the -1 is to remove trailing empty line
                vals_array = np.append(vals_array, values)
            vals_array = vals_array.astype(np.float)
            vals_array = vals_array.reshape((len(files_list), -1))
            vals_dict[vals] = vals_array
        
        return vals_dict

class read_kcap_covariance_theory(kcap_deriv):
    def __init__(self, mock_run):
        self.kids_mocks_dir = os.environ['KIDS_MOCKS_DIR']
        self.kids_mocks_root_name = os.environ['KIDS_MOCKS_ROOT_NAME']
        self.mock_run = self.check_mock_run_exists(mock_run)

    def read_covariance(self):
        covariance_file = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/theory_data_covariance/covariance.txt'
        inv_covariance_file = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/theory_data_covariance/inv_covariance.txt'

        covariance = np.loadtxt(covariance_file, skiprows = 1)

        inv_covariance = np.loadtxt(inv_covariance_file, skiprows = 1)

        return covariance, inv_covariance
    
    def read_theory(self):
        theory_file = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/theory_data_covariance/theory.txt'
        theory = np.loadtxt(theory_file, skiprows = 1)
        return theory

    def read_theory_data(self):
        data_file = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/theory_data_covariance/data.txt'
        data = np.loadtxt(data_file, skiprows = 1)
        return data



def run_kcap_deriv(mock_run, param_to_vary, params_to_fix, vals_to_diff, step_size):
    kcap_run = kcap_deriv(mock_run = mock_run, 
                          param_to_vary = param_to_vary, 
                          params_to_fix = params_to_fix,
                          vals_to_diff = vals_to_diff)
    check = kcap_run.check_existing_derivs()
    if check is True:
        exit()
    else:
        print("Not all values found, continuing script...")
        pass
    params = kcap_run.get_params()
    abs_step_size = kcap_run.write_deriv_values(step_size = step_size)
    kcap_run.run_deriv_kcap(mpi_opt = True, threads = 4)
    kcap_run.copy_deriv_vals_to_mocks(step_size = step_size, abs_step_size = abs_step_size)
    kcap_run.first_deriv(abs_step_size = abs_step_size)
    kcap_run.cleanup()

def get_values(mock_run, vals_to_read):
    values_method = read_kcap(mock_run = mock_run, vals_to_read = vals_to_read)
    values_read = values_method.read_vals()
    return values_read

def get_covariance(mock_run):
    values_method = read_kcap_covariance_theory(mock_run = mock_run)
    covariance, inv_covariance = values_method.read_covariance()
    return covariance, inv_covariance

def get_fiducial_means(mock_run):
    values_method = read_kcap_covariance_theory(mock_run = mock_run)
    fiducial_means = values_method.read_theory()
    return fiducial_means

if __name__ == "__main__":
    # run_kcap_deriv(mock_run = 0, 
    #                param_to_vary = "cosmological_parameters--omch2", 
    #                params_to_fix = ["cosmological_parameters--sigma_8", "intrinsic_alignment_parameters--a"],
    #                vals_to_diff = ["shear_xi_minus", "shear_xi_plus"],
    #                step_size = 0.02)
    # temp_vals = get_values(mock_run = 0, vals_to_read = ["shear_xi_plus", "shear_xi_minus"])
    # print(temp_vals)
    covariance, inv_covariance = get_covariance(mock_run = 0)
    print(len(covariance))
    print(len(covariance[0]))