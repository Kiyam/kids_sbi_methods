import numpy as np
import configparser as cfg
import subprocess
import os
import glob
import tarfile
import shutil
import errno
import re
import time
from environs import Env

#NOTE - The params that can be varied in the future are h, ns, A_IA, A_bary, sigma_z, sigma_c, sigma_8

class kcap_deriv:
    def __init__(self, mock_run, param_to_vary, params_to_fix, vals_to_diff,
                       mocks_dir = None, mocks_name = None, mocks_ini_file = None, mocks_values_file = None,
                       deriv_dir = None, deriv_name = None, deriv_ini_file = None, deriv_values_file = None, deriv_values_list_file = None):
        """
        Gets variable from .env file
        """

        env = Env()
        env.read_env()

        self.cosmosis_src_dir = env.str('cosmosis_src_dir')

        # mocks_dir settings
        if mocks_dir == None:
            self.kids_mocks_dir = env.str('kids_mocks_dir')
        else:
            self.kids_mocks_dir = mocks_dir

        # mocks_name settings
        if mocks_name == None:
            self.kids_mocks_root_name = env.str('kids_mocks_root_name')
        else:
            self.kids_mocks_root_name = mocks_name
        
        # mocks ini file settings
        if mocks_ini_file == None:
            self.kids_pipeline_ini_file = env.str('kids_pipeline_ini_file')
        else:
            self.kids_pipeline_ini_file = mocks_ini_file

        # mocks values file settings
        if mocks_values_file == None:
            self.kids_pipeline_values_file = env.str('kids_pipeline_values_file')
        else:
            self.kids_pipeline_values_file = mocks_values_file

        # deriv_dir settings
        if deriv_dir == None:
            self.kids_deriv_dir = env.str('kids_deriv_dir')
        else:
            self.kids_deriv_dir = deriv_dir
        
        # deriv output filename settings
        if deriv_name == None:
            self.kids_deriv_root_name = env.str('kids_deriv_root_name')
        else:
            self.kids_deriv_root_name = deriv_name
        
        # deriv ini file settings
        if deriv_ini_file == None:
            self.kids_deriv_ini_file = env.str('kids_deriv_ini_file')
        else:
            self.kids_deriv_ini_file = deriv_ini_file

        # deriv values file settings
        if deriv_values_file == None:
            self.kids_deriv_values_file = env.str('kids_deriv_values_file')
        else:
            self.kids_deriv_values_file = deriv_values_file
        
        # deriv values list file settings
        if deriv_values_list_file == None:
            self.kids_deriv_values_list = env.str('kids_pipeline_deriv_values_list')
        else:
            self.kids_deriv_values_list = deriv_values_list_file
        
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
        if type(mock_run) is not str:
            mock_run = str(mock_run)
        elif type(mock_run) is str:
            pass
        else:
            raise Exception("Sorry, the requested mock run, %s, is named incorrectly" % mock_run)

        if os.path.exists(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+mock_run):
            pass
        elif os.path.exists(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+mock_run+'.tgz'):
            print("Need to untar files first...")
            self.extract_tar(mock_run = mock_run)
            print("Succesfully untarred")
        else:
            raise Exception("Sorry, the requested mock run, %s, doesn't exist" % mock_run)
        return mock_run

    def extract_tar(self, mock_run, option = "mocks", stencil_pts = 5):
        """
        Untars file
        """
        if option == "mocks":
            with tarfile.open(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+mock_run+'.tgz', 'r:gz') as kcap_tar:
                kcap_tar.extractall(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+mock_run)
                kcap_tar.close()
                print("Mock run extracted is: %s" % mock_run)
        elif option == "derivs":
            for step in range(stencil_pts - 1):
                with tarfile.open(self.kids_deriv_dir+'/'+self.kids_deriv_root_name+'_'+str(step)+'.tgz') as kcap_tar:
                    kcap_tar.extractall(self.kids_deriv_dir+'/'+self.kids_deriv_root_name+'_'+str(step))
                    kcap_tar.close()
        else:
            raise Exception("Ill defined option passed as argument to extract_tar method")      

    def check_ini_settings(self, ini_file_to_check):
        """
        Modifies the deriv_values_list.ini file
        """
        values_config = cfg.ConfigParser()

        if ini_file_to_check == 'pipeline_file':
            to_change = 0
            values_config.read(self.kids_pipeline_ini_file)
            if values_config['DEFAULT']['RESULTS_PATH'] == self.kids_mocks_dir:
                pass
            else:
                values_config['DEFAULT']['RESULTS_PATH'] = self.kids_mocks_dir
                to_change += 1
                
            if values_config['DEFAULT']['RESULTS_NAME'] == self.kids_mocks_root_name:
                pass
            else:
                values_config['DEFAULT']['RESULTS_NAME'] = self.kids_mocks_root_name
                to_change += 1

            if to_change > 0:
                print("Setting a few pipeline ini file values...")
                with open(self.kids_deriv_values_file, 'w') as configfile:
                    values_config.write(configfile)
        
        elif ini_file_to_check == 'deriv_file':
            to_change = 0
            values_config.read(self.kids_deriv_ini_file)
            if values_config['DEFAULT']['RESULTS_PATH'] == self.kids_deriv_dir:
                pass
            else:
                values_config['DEFAULT']['RESULTS_PATH'] = self.kids_deriv_dir
                to_change += 1
                
            if values_config['DEFAULT']['RESULTS_NAME'] == self.kids_deriv_root_name:
                pass
            else:
                values_config['DEFAULT']['RESULTS_NAME'] = self.kids_deriv_root_name
                to_change += 1

            if to_change > 0:
                print("Setting a few deriv pipeline ini file values...")
                with open(self.kids_deriv_ini_file, 'w') as configfile:
                    values_config.write(configfile)

    def get_params(self):
        """
        Gets parameters from the specified mock run
        """
        parameter_dict = {} # Dictionary of parameters to write to file
        for item in self.parameter_list:
            header, name = item.split("--")
            if name == "sigma_8":
                name = "sigma_8_input"
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

    def write_deriv_values(self, step_size, stencil_pts = 5):
        """
        Modifies the deriv_values_list.ini file
        """
        values_config = cfg.ConfigParser()
        values_config.read(self.kids_deriv_values_file)

        for param in self.params_to_fix:
            if param in self.param_dict:
                header, name = param.split("--")
                if name == "sigma_8":
                    name = "sigma_8_input"
                elif name == "s_8":
                    name = "s_8_input"
                values_config[header][name] = str(self.param_dict[param])
            else:
                raise Exception("Unknown parameter specified in params_to_fix")
        
        dx_array = np.arange(stencil_pts)
        middle_index = int((stencil_pts - 1)/2)
        dx_array = dx_array - dx_array[middle_index]
        dx_array = np.delete(dx_array, middle_index)

        if self.param_to_vary in self.param_dict:
            if self.param_name == "sigma_8":
                name = "sigma_8_input"
            elif self.param_name == "s_8":
                name = "s_8_input"
            else:
                name = self.param_name
                
            middle_val = self.param_dict[self.param_to_vary]
            if middle_val == 0.:
                abs_step_size = step_size
            else:
                abs_step_size = np.absolute(middle_val) * step_size
            vals_array = middle_val + dx_array*abs_step_size

            new_param_string = str(vals_array[0]) + " " + str(middle_val) + " " + str(vals_array[-1])       
            values_config[self.param_header][name] = new_param_string
            file_text = ["#"+self.param_header+"--"+name+"\n"]
            for val in vals_array:
                file_text.append(str(val) + "\n")

            values_list_file = open(self.kids_deriv_values_list, "w")
            values_list_file.writelines(file_text)
            values_list_file.close()

        else:
            raise Exception("Badly defined parameter to vary...")

        with open(self.kids_deriv_values_file, 'w') as configfile:
            values_config.write(configfile)
        
        return step_size, abs_step_size

    def run_deriv_kcap(self, mpi_opt = False, cluster = True):
        self.check_ini_settings(ini_file_to_check = 'deriv_file')
        if mpi_opt == True:
            subprocess.run(["mpirun", "-n" , "12", "--use-hwthread-cpus", "cosmosis", "--mpi", self.kids_deriv_ini_file])
        elif cluster == True:
            subprocess.run(["sbatch", "/share/splinter/klin/slurm/slurm_kcap_derivs.sh"])
        elif mpi_opt == False:
            subprocess.run(["cosmosis", self.kids_deriv_ini_file])
        else: 
            raise Exception
    
    def poll_cluster_finished(self, stencil_pts = 5):
        start_time = time.time()
        elapsed = time.time() - start_time
        finished = False
        while elapsed <= 3600. and finished != True:
            if len(glob.glob(self.kids_deriv_dir+'/'+self.kids_deriv_root_name+'_*.tgz')) < stencil_pts - 1:
                time.sleep(15)
            elif len(glob.glob(self.kids_deriv_dir+'/'+self.kids_deriv_root_name+'_*.tgz')) == stencil_pts - 1:
                finished = True
        print("Waiting to ensure all IO operations are finished")
        time.sleep(40)

    def copy_deriv_vals_to_mocks(self, step_size, abs_step_size, stencil_pts = 5):
        if len(glob.glob(self.kids_deriv_dir+'/'+self.kids_deriv_root_name+'_*/')) == stencil_pts - 1:
            pass
        elif len(glob.glob(self.kids_deriv_dir+'/'+self.kids_deriv_root_name+'_*.tgz')) == stencil_pts - 1:
            print("Need to untar files first...")
            self.extract_tar(self.mock_run, option = "derivs", stencil_pts = stencil_pts)
            print("Succesfully untarred")
        else:
            raise Exception("Sorry, you seem to be missing the derivatives, try running KCAP for the requested derivative steps first")
        
        if stencil_pts == 3:
            step_list = ["-1dx", "+1dx"]
        elif stencil_pts == 5:
            step_list = ["-2dx", "-1dx", "+1dx", "+2dx"]
        elif stencil_pts == 7:
            step_list = ["-3dx", "-2dx", "-1dx", "+1dx", "+2dx", "+3dx"]
        elif stencil_pts == 9:
            step_list = ["-4dx", "-3dx", "-2dx", "-1dx", "+1dx", "+2dx", "+3dx", "+4dx"]
        else:
            raise Exception("Invalid stencil number inputted")

        stepsize_file = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/deriv_stepsizes/'+self.param_to_vary+"_stepsize.txt"
        if not os.path.exists(os.path.dirname(stepsize_file)):
            try:
                os.makedirs(os.path.dirname(stepsize_file))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        with open(stepsize_file, "w") as f:
            f.write(self.param_to_vary+"_relative_stepsize="+str(step_size)+"\n"+self.param_to_vary+"_absolute_stepsize="+str(abs_step_size))

        for deriv_run in range(stencil_pts - 1):
            for param in self.vals_to_diff:
                if "theory" or "covariance" in param:
                    new_subdir_root = "theory_data_covariance_" + self.param_to_vary
                    shutil.copytree(self.kids_deriv_dir+'/'+self.kids_deriv_root_name+'_'+str(deriv_run)+'/theory_data_covariance', 
                                    self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+new_subdir_root+step_list[deriv_run])
                else:
                    new_subdir_root = param + "_" + self.param_to_vary
                    shutil.copytree(self.kids_deriv_dir+'/'+self.kids_deriv_root_name+'_'+str(deriv_run)+'/'+param, 
                                    self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+new_subdir_root+step_list[deriv_run])
    
    def check_existing_derivs(self):
        print("Checking if the corresponding derivatives exist...")
        check_count = 0
        for deriv_vals in self.vals_to_diff:
            if "covariance" in deriv_vals:
                if os.path.exists(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_to_vary+'_deriv/covariance.txt'):
                    print("Files for %s numerical derivative values wrt to %s found." % (deriv_vals, self.param_to_vary))
                    check_count += 1
                else:
                    print("Missing derivatives for %s wrt to %s." % (deriv_vals, self.param_to_vary))
            elif "theory" in deriv_vals:
                if os.path.exists(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_to_vary+'_deriv/theory.txt'):
                    print("Files for %s numerical derivative values wrt to %s found." % (deriv_vals, self.param_to_vary))
                    check_count += 1
                else:
                    print("Missing derivatives for %s wrt to %s." % (deriv_vals, self.param_to_vary))
            else:      
                num_bins = len(glob.glob(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'/bin*.txt'))
                num_found_bins = len(glob.glob(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_to_vary+'_deriv/bin*.txt'))
                if num_found_bins == num_bins:
                    print("Files for %s numerical derivative values wrt to %s found." % (deriv_vals, self.param_to_vary))
                    check_count += 1
                else:
                    print("Missing derivatives for %s wrt to %s." % (deriv_vals, self.param_to_vary))
        
        stepsize_file = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/deriv_stepsizes/'+self.param_to_vary+"_stepsize.txt"
        
        if os.path.exists(os.path.dirname(stepsize_file)):
            print("Stepsize file for deriv wrt to %s found" % self.param_to_vary)
            check_count += 1
        else:
            print("Missing stepsize file for %s derivatives" % self.param_to_vary)
        
        if check_count == len(self.vals_to_diff) + 1:
            print("All wanted numerical derivative values found!")
            return True
        else:
            return False

    def cleanup_deriv_folder(self):
        print("Checking all files exist as expected and cleaning up...")
        files_found = self.check_existing_derivs()
        if files_found is True:
            print("Initiating cleanup...")
            shutil.rmtree(self.kids_deriv_dir)
            os.makedirs(self.kids_deriv_dir)
            check_dir = os.listdir(self.kids_deriv_dir)
            
            if len(check_dir) == 0:
                if len(glob.glob(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/*_'+self.param_to_vary+'*dx')) == 0:
                    print("Derivatives temprorary files succesfully cleaned up.")
            else:
                raise Exception("Error during directory cleanup, please manually inspect!")
        else:
            print("Not all files found, exiting cleanup. Please manually inspect!")
            exit()
    
    def cleanup_dx(self, stencil_pts = 5):
        for deriv_vals in self.vals_to_diff:
            if "covariance" or "theory" in deriv_vals:
                if stencil_pts >= 3:
                    shutil.rmtree(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/theory_data_covariance_'+self.param_to_vary+'-1dx/')
                    shutil.rmtree(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/theory_data_covariance_'+self.param_to_vary+'+1dx/')
                if stencil_pts >= 5:
                    shutil.rmtree(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/theory_data_covariance_'+self.param_to_vary+'-2dx/')
                    shutil.rmtree(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/theory_data_covariance_'+self.param_to_vary+'+2dx/')
                if stencil_pts >= 7:
                    shutil.rmtree(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/theory_data_covariance_'+self.param_to_vary+'-3dx/')
                    shutil.rmtree(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/theory_data_covariance_'+self.param_to_vary+'+3dx/')
                if stencil_pts >= 9:
                    shutil.rmtree(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/theory_data_covariance_'+self.param_to_vary+'-4dx/')
                    shutil.rmtree(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/theory_data_covariance_'+self.param_to_vary+'+4dx/')
            else:
                if stencil_pts >= 3:
                    shutil.rmtree(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_to_vary+'-1dx/')
                    shutil.rmtree(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_to_vary+'+1dx/')
                if stencil_pts >= 5:
                    shutil.rmtree(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_to_vary+'-2dx/')
                    shutil.rmtree(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_to_vary+'+2dx/')
                if stencil_pts >= 7:
                    shutil.rmtree(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_to_vary+'-3dx/')
                    shutil.rmtree(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_to_vary+'+3dx/')
                if stencil_pts >= 9:
                    shutil.rmtree(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_to_vary+'-4dx/')
                    shutil.rmtree(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_to_vary+'+4dx/') 

    def first_deriv(self, abs_step_size, stencil_pts = 5):
        """
        Calculates the first derivative using a 5 point stencil
        """
        
        if stencil_pts == 3:
            step_list = ["-1dx", "+1dx"]
            stencil_coeffs = np.array([-1/2, 1/2])
        elif stencil_pts == 5:
            step_list = ["-2dx", "-1dx", "+1dx", "+2dx"]
            stencil_coeffs = np.array([1/12, -2/3, 2/3, -1/12])
        elif stencil_pts == 7:
            step_list = ["-3dx", "-2dx", "-1dx", "+1dx", "+2dx", "+3dx"]
            stencil_coeffs = np.array([-1/60, 3/20, -3/4, 3/4, -3/20, 1/60])
        elif stencil_pts == 9:
            step_list = ["-4dx", "-3dx", "-2dx", "-1dx", "+1dx", "+2dx", "+3dx", "+4dx"]
            stencil_coeffs = np.array([1/280, -4/105, 1/5, -4/5, 4/5, -1/5, 4/105, -1/280])
        else:
            raise Exception("Invalid stencil number inputted")

        for deriv_vals in self.vals_to_diff:
            if "covariance" in deriv_vals:
                file_root = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/theory_data_covariance_'+self.param_to_vary
                dx_files = [file_root + dx_step + '/covariance.txt' for dx_step in step_list]
                dx_vals = np.asarray([np.genfromtxt(file_name) for file_name in dx_files])
                cov_shape = dx_vals.shape[1]
                dx_vals = dx_vals.reshape(dx_vals.shape[0], -1)

            elif "theory" in deriv_vals:
                file_root = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/theory_data_covariance_'+self.param_to_vary
                dx_files = [file_root + dx_step + '/theory.txt' for dx_step in step_list]
                dx_vals = np.asarray([np.genfromtxt(file_name) for file_name in dx_files])

            else:
                # Old binned method of getting the derivatives
                file_root = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_to_vary
                dx_files = [glob.glob(file_root + dx_step + '/bin*.txt') for dx_step in step_list]
                
                len_first = len(dx_files[0]) if dx_files else None
                assert all(len(i) == len_first for sub_list in dx_files), "Some dx stepsize files missing."
                
                #fetch bin names
                bin_names = [bin_name.split("/")[-1].replace(".txt", "") for bin_name in dx_files[0]]
                
                dx_vals = []
                for dx_bin_list in dx_files:
                    temp_dx_vals = np.array([])
                    for dx_file_name in dx_bin_list:
                        values = np.genfromtxt(dx_file_name)
                        temp_dx_vals = np.append(temp_dx_vals, values)
                    dx_vals.append(temp_dx_vals)
                dx_vals = np.asarray(dx_vals)                    

            print("All values needed for %s derivatives wrt to %s found, calculating and saving to file..." % (deriv_vals, self.param_to_vary))            
            
            first_deriv_vals = np.dot(stencil_coeffs, dx_vals)/abs_step_size

            deriv_dir_path = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_to_vary+"_deriv/"

            if not os.path.exists(os.path.dirname(deriv_dir_path)):
                try:
                    os.makedirs(os.path.dirname(deriv_dir_path))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise

            if "covariance" in deriv_vals:
                deriv_file = deriv_dir_path+"covariance.txt"
                first_deriv_vals = first_deriv_vals.reshape(cov_shape, cov_shape)
                np.savetxt(deriv_file, first_deriv_vals, newline="\n", header="covariance")
            elif "theory" in deriv_vals:
                deriv_file = deriv_dir_path+"theory.txt"
                np.savetxt(deriv_file, first_deriv_vals, newline="\n", header="theory")
            else:
                first_deriv_vals = first_deriv_vals.reshape((len(bin_names), -1))
                for i, vals in enumerate(first_deriv_vals):
                    deriv_file = deriv_dir_path+bin_names[i]+".txt"
                    np.savetxt(deriv_file, vals, newline="\n", header=bin_names[i])
        print("Derivatives saved succesfully")

    def first_omega_m_deriv(self, params_varied):       
        h = self.read_param_from_txt_file(file_location = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/cosmological_parameters/values.txt', parameter = 'h0')
        ombh2 = self.read_param_from_txt_file(file_location = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/cosmological_parameters/values.txt', parameter = 'ombh2')
        omch2 = self.read_param_from_txt_file(file_location = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/cosmological_parameters/values.txt', parameter = 'omch2')
        omega_m = self.read_param_from_txt_file(file_location = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/cosmological_parameters/values.txt', parameter = 'omega_m')
        omnuh2 = self.read_param_from_txt_file(file_location = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/cosmological_parameters/values.txt', parameter = 'omnuh2')
        
        for deriv_vals in self.vals_to_diff:
            if "covariance" in deriv_vals:
                if "cosmological_parameters--omch2" in params_varied: 
                    omch_deriv_file = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/theory_cosmological_parameters--omch2_deriv/covariance.txt'
                    with open(omch_deriv_file, 'r') as flx:
                        omch_deriv = np.loadtxt(flx)
                if "cosmological_parameters--ombh2" in params_varied:
                    ombh_deriv_file = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/theory_cosmological_parameters--ombh2_deriv/covariance.txt'
                    with open(ombh_deriv_file, 'r') as flx:
                        ombh_deriv = np.loadtxt(flx)
                if "cosmological_parameters--h0" in params_varied:
                    h0_deriv_file = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/theory_cosmological_parameters--h0_deriv/covariance.txt'
                    with open(h0_deriv_file, 'r') as flx:
                        h0_deriv = np.loadtxt(flx)
            elif "theory" in deriv_vals:
                if "cosmological_parameters--omch2" in params_varied: 
                    omch_deriv_file = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/theory_cosmological_parameters--omch2_deriv/theory.txt'
                    with open(omch_deriv_file, 'r') as flx:
                        omch_deriv = np.loadtxt(flx)
                if "cosmological_parameters--ombh2" in params_varied:
                    ombh_deriv_file = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/theory_cosmological_parameters--ombh2_deriv/theory.txt'
                    with open(ombh_deriv_file, 'r') as flx:
                        ombh_deriv = np.loadtxt(flx)
                if "cosmological_parameters--h0" in params_varied:
                    h0_deriv_file = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/theory_cosmological_parameters--h0_deriv/theory.txt'
                    with open(h0_deriv_file, 'r') as flx:
                        h0_deriv = np.loadtxt(flx)
            else:
                if "cosmological_parameters--omch2" in params_varied: 
                    omch_deriv_files = glob.glob(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_cosmological_parameters--omch2_deriv/bin*.txt')
                    bin_names = [bin_name.split("/")[-1].replace(".txt", "") for bin_name in omch_deriv_files]
                    omch_deriv = np.array([])
                    for dx_file_name in omch_deriv_files:
                        with open(dx_file_name, 'r') as flx:
                            values = np.loadtxt(flx)
                        omch_deriv = np.append(omch_deriv, values)
                    omch_deriv = omch_deriv.reshape((len(bin_names), -1))
                if "cosmological_parameters--ombh2" in params_varied: 
                    ombh_deriv_files = glob.glob(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_cosmological_parameters--ombh2_deriv/bin*.txt')
                    bin_names = [bin_name.split("/")[-1].replace(".txt", "") for bin_name in ombh_deriv_files]
                    ombh_deriv = np.array([])
                    for dx_file_name in ombh_deriv_files:
                        with open(dx_file_name, 'r') as flx:
                            values = np.loadtxt(flx)
                        ombh_deriv = np.append(ombh_deriv, values)
                    ombh_deriv = ombh_deriv.reshape((len(bin_names), -1))
                if "cosmological_parameters--h0" in params_varied: 
                    h0_deriv_files = glob.glob(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_cosmological_parameters--h0_deriv/bin*.txt')
                    bin_names = [bin_name.split("/")[-1].replace(".txt", "") for bin_name in h0_deriv_files]
                    h0_deriv = np.array([])
                    for dx_file_name in h0_deriv_files:
                        with open(dx_file_name, 'r') as flx:
                            values = np.loadtxt(flx)
                        h0_deriv = np.append(h0_deriv, values)
                    h0_deriv = h0_deriv.reshape((len(bin_names), -1))
            try:
                omch2_component = omch_deriv * (h**2)
            except:
                omch2_component = 0
            try:
                ombh2_component = ombh_deriv * (h**2)
            except:
                ombh2_component = 0
            try:
                # h_component = -0.5 * ((omch2 + ombh2 + omnuh2)**0.5) * (omega_m ** (-1.5)) * h0_deriv
                # h_component = -0.5 * (omega_m ** (-0.5)) * (h ** 2) * h0_deriv
                h_component = - (h ** 3)/(2 * (omch2 + ombh2 + omnuh2)) * h0_deriv
            except:
                h_component = 0
                
            omega_m_deriv_vals = omch2_component + ombh2_component + h_component
                
            print("All values needed for %s derivatives wrt to omega_m calculated, saving to file..." % (deriv_vals))

            # if "covariance" or "theory" in deriv_vals:
            #     deriv_dir_path = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/theory_data_covariance_cosmological_parameters--omega_m_deriv/'
            # else:
            deriv_dir_path = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_cosmological_parameters--omega_m_deriv/'

            if not os.path.exists(os.path.dirname(deriv_dir_path)):
                try:
                    os.makedirs(os.path.dirname(deriv_dir_path))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise

            if "covariance" in deriv_vals:
                deriv_file = deriv_dir_path+"covariance.txt"
                np.savetxt(deriv_file, omega_m_deriv_vals, newline="\n", header="covariance")
            elif "theory" in deriv_vals:
                deriv_file = deriv_dir_path+"theory.txt"
                np.savetxt(deriv_file, omega_m_deriv_vals, newline="\n", header="theory")
            else:
                for i, vals in enumerate(omega_m_deriv_vals):
                    deriv_file = deriv_dir_path+bin_names[i]+".txt"
                    np.savetxt(deriv_file, vals, newline="\n", header=bin_names[i])
            print("Derivatives saved succesfully")

class kcap_delfi(kcap_deriv):
    def __init__(self, params_to_vary, mocks_dir = None, mocks_name = None, mocks_ini_file = None, mocks_values_file = None,
                 ombh2_fiducial = 0.0224,
                 h0_fiducial = 0.674,
                 folders_to_keep = ["cosmological_parameters",
                                    "intrinsic_alignment_parameters",
                                    "growth_parameters",
                                    "bias_parameters",
                                    "halo_model_parameters",
                                    "likelihoods",
                                    "theory_data_covariance"], 
                 files_to_remove = ["theory_data_covariance/covariance.txt", "theory_data_covariance/inv_covariance.txt"],
                 verbose = False,
                 comm = None):
        """
        Gets variable from .env file
        """

        env = Env()
        env.read_env()

        self.cosmosis_src_dir = env.str('cosmosis_src_dir')
        
        # mocks_dir settings
        if mocks_dir == None:
            self.kids_mocks_dir = env.str('kids_mocks_dir')
        else:
            self.kids_mocks_dir = mocks_dir

        # mocks_name settings
        if mocks_name == None:
            self.kids_mocks_root_name = env.str('kids_mocks_root_name')
        else:
            self.kids_mocks_root_name = mocks_name
        
        # mocks ini file settings
        if mocks_ini_file == None:
            self.kids_pipeline_ini_file = env.str('kids_pipeline_ini_file')
        else:
            self.kids_pipeline_ini_file = mocks_ini_file

        # mocks values file settings
        if mocks_values_file == None:
            self.kids_pipeline_values_file = env.str('kids_pipeline_values_file')
        else:
            self.kids_pipeline_values_file = mocks_values_file
        
        if isinstance(params_to_vary, list):
            self.params_to_vary = params_to_vary
        else:
            raise Exception("param_to_vary variable must be a string")
        
        self.ombh2_fiducial = ombh2_fiducial
        self.h0_fiducial = h0_fiducial
        
        self.verbose = verbose
        self.folders_to_keep = folders_to_keep
        self.files_to_remove = files_to_remove
        
        if comm:
            self.comm = comm
        else:
            self.comm = None
    
    def run_delfi_kcap(self, rank):
        pipeline_file = self.kids_pipeline_ini_file + '_' + str(rank) + '.ini'
        subprocess.run(["cosmosis", pipeline_file])
    
    def set_delfi_kcap_pipeline(self, rank):      
        pipeline_file = self.kids_pipeline_ini_file + '_' + str(rank) + '.ini'
        pipeline_config = cfg.ConfigParser()
        pipeline_config.read(pipeline_file)
        
        to_change = 0
        
        if pipeline_config['DEFAULT']['RESULTS_PATH'] != self.kids_mocks_dir:
            pipeline_config['DEFAULT']['RESULTS_PATH'] = self.kids_mocks_dir
            to_change += 1
            
        if pipeline_config['DEFAULT']['RESULTS_NAME'] != self.kids_mocks_root_name + '_' + str(rank):
            pipeline_config['DEFAULT']['RESULTS_NAME'] = self.kids_mocks_root_name + '_' + str(rank)
            to_change += 1
        
        if pipeline_config['pipeline']['values'] != self.mocks_values_file + '_' + str(rank):
            pipeline_config['pipeline']['values'] = self.mocks_values_file + '_' + str(rank)
            to_change += 1
            
        with open(pipeline_file, 'w') as configfile:
            pipeline_config.write(configfile)
    
    def write_delfi_kcap_values(self, theta, rank):
        """
        Modifies the deriv_values_list.ini file
        """
        assert len(theta) == len(self.params_to_vary), "Different number of parameters to vary defined vs parameter value ranges given."
        values_config = cfg.ConfigParser()
        pipeline_values_file = self.kids_pipeline_values_file + '_' + str(rank) + '.ini'
        values_config.read(pipeline_values_file)

        for i, param in enumerate(self.params_to_vary):
            header, name = param.split("--")
            if name == "sigma_8":
                name = "sigma_8_input"
            elif name == "s_8":
                name = "s_8_input"
            elif name == "omega_m":
                ### This is the complicated one, need to find appropriate omch2 values
                omch2 = theta[i] * (self.h0_fiducial ** 2) - self.ombh2_fiducial
                name = "omch2"
            if values_config.has_option(header, name):
                values_config[header][name] = str(theta[i])
            else:
                raise Exception("Unknown parameter specified in params_to_fix")

        with open(pipeline_values_file, 'w') as configfile:
            values_config.write(configfile)
    
    def delete_delfi_unwanted(self, rank):
        #! This should only be run on the direct kcap output, not any folder containing derivatives
        all_folders = glob.glob(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+str(rank)+'/*')
        kept_folders = []
        for folder in all_folders:
            for to_keep in self.folders_to_keep:
                if re.search(to_keep, folder):
                    kept_folders.append(folder)
        
        folders_to_delete = list(set(all_folders) - set(kept_folders))
        
        for folder_to_delete in folders_to_delete:
            shutil.rmtree(folder_to_delete)
        
        for file_to_remove in self.files_to_remove:
            try:
                os.remove(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+str(rank)+'/'+file_to_remove)
            except:
                pass
            
        self.check_wanted_folders(rank = rank)
        if self.verbose == True:
            print("Succesfully deleted all superfluous files and folders in extracted mock run.")
    
    def check_wanted_folders(self, rank):
        all_folders = glob.glob(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+str(rank)+'/*')
        if len(all_folders) != len(self.folders_to_keep):
            raise Exception("Some folders missing! Please double check the files carefully")
    
    def return_data_vector(self, rank):
        theory_file = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+str(rank)+'/theory_data_covariance/theory.txt'
        theory = np.loadtxt(theory_file, skiprows = 1)
        return theory

    def simulate(self, theta):
        if self.comm:
            rank = self.comm.Get_rank()
        else:
            rank = 0
        if self.verbose == True:
            print('Running simulator on MPI rank %s' %(rank))
            
        self.write_delfi_kcap_values(theta, rank)
        self.run_delfi_kcap(rank)
        self.delete_delfi_unwanted(rank)
        data_vector = self.return_data_vector(rank)
        return data_vector
    
class organise_kcap_output(kcap_deriv):
    def __init__(self, mock_run_start, num_mock_runs, mocks_dir = None, mocks_name = None,
                 folders_to_keep = ["shear_xi_minus_binned", 
                                    "shear_xi_plus_binned", 
                                    "cosmological_parameters",
                                    "intrinsic_alignment_parameters",
                                    "growth_parameters",
                                    "bias_parameters",
                                    "halo_model_parameters",
                                    "likelihoods",
                                    "theory_data_covariance"], 
                 files_to_remove = ["theory_data_covariance/covariance.txt", "theory_data_covariance/inv_covariance.txt"]):
        env = Env()
        env.read_env()
        
        self.mock_run_start = mock_run_start
        self.num_mock_runs = num_mock_runs
        # mocks_dir settings
        if mocks_dir == None:
            self.kids_mocks_dir = env.str('kids_mocks_dir')
        else:
            self.kids_mocks_dir = mocks_dir

        # mocks_name settings
        if mocks_name == None:
            self.kids_mocks_root_name = env.str('kids_mocks_root_name')
        else:
            self.kids_mocks_root_name = mocks_name
        
        self.folders_to_keep = folders_to_keep
        self.files_to_remove = files_to_remove
    
    def extract_all_runs(self):
        for i in range(self.num_mock_runs):
            mock_run = self.mock_run_start + i
            self.check_mock_run_exists(mock_run)
        print("All files found and extracted!")
    
    def extract_and_delete(self):
        for i in range(self.num_mock_runs):
            mock_run = self.mock_run_start + i
            self.check_mock_run_exists(mock_run)
            self.delete_unwanted(mock_run)
            self.delete_mock_tgz(mock_run)

    def delete_unwanted(self, mock_run):
        #! This should only be run on the direct kcap output, not any folder containing derivatives
        all_folders = glob.glob(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+str(mock_run)+'/*')
        kept_folders = []
        for folder in all_folders:
            for to_keep in self.folders_to_keep:
                if re.search(to_keep, folder):
                    kept_folders.append(folder)
        
        folders_to_delete = list(set(all_folders) - set(kept_folders))
        
        for folder_to_delete in folders_to_delete:
            shutil.rmtree(folder_to_delete)
        
        for file_to_remove in self.files_to_remove:
            try:
                os.remove(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+str(mock_run)+'/'+file_to_remove)
            except:
                pass
            
        self.check_wanted_folders(mock_run = mock_run)
        print("Succesfully deleted all superfluous files and folders in extracted mock run %s." % str(mock_run))
    
    def delete_all_unwanted(self):
        #! This should only be run on the direct kcap output, not any folder containing derivatives
        for i in range(self.num_mock_runs):
            mock_run = int(self.mock_run_start) + i
            self.delete_unwanted(mock_run = mock_run)
            print("All desired files and folders in extracted mock run %s have been succesfully deleted." % str(mock_run))
        print("All unwanted folders and files delete with hopefully no missing files.")
        
    def check_wanted_folders(self, mock_run):
        all_folders = glob.glob(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+str(mock_run)+'/*')
        if len(all_folders) != len(self.folders_to_keep):
            raise Exception("Some folders missing! Please double check the files in mock run %s carefully" % str(mock_run))
    
    def delete_mock_tgz(self, mock_run):
        tgz_to_delete = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+str(mock_run)+'.tgz'
        os.remove(tgz_to_delete)
        print("Removed %s" %str(tgz_to_delete))
        
    def delete_all_tgz(self):
        all_tgz = glob.glob(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'*.tgz')
        for tgz_file in all_tgz:
            os.remove(tgz_file)
            print("Removed %s" %str(tgz_file))
        print("All .tgz files removed!")

class read_kcap_values(kcap_deriv):
    def __init__(self, mock_run, mocks_dir = None, mocks_name = None, 
                        bin_order = None):

        env = Env()
        env.read_env()

        # mocks_dir settings
        if mocks_dir == None:
            self.kids_mocks_dir = env.str('kids_mocks_dir')
        else:
            self.kids_mocks_dir = mocks_dir

        # mocks_name settings
        if mocks_name == None:
            self.kids_mocks_root_name = env.str('kids_mocks_root_name')
        else:
            self.kids_mocks_root_name = mocks_name

        self.mock_run = self.check_mock_run_exists(mock_run)

        if bin_order == None:
            self.bin_order = ['bin_1_1', 'bin_2_1', 'bin_3_1', 'bin_4_1', 'bin_5_1',
                              'bin_2_2', 'bin_3_2', 'bin_4_2', 'bin_5_2',
                              'bin_3_3', 'bin_4_3', 'bin_5_3',
                              'bin_4_4', 'bin_5_4',
                              'bin_5_5']
        else:
            self.bin_order = bin_order

    def read_vals(self, vals_to_read):
        if isinstance(vals_to_read, list):
            vals_to_read = vals_to_read
        elif isinstance(vals_to_read, str):
            vals_to_read = [vals_to_read]
        else:
            raise Exception("Badly defined values to read, needs to be of type string or list")

        vals_dict = {}
        for val_names in vals_to_read:
            if "theory" in val_names:
                if os.path.exists(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+val_names+'/theory.txt'):
                    vals_file = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+val_names+'/theory.txt'
                    vals_array = np.genfromtxt(vals_file, comments = '#')
                elif "theory" == val_names:
                    vals_file = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/theory_data_covariance/theory.txt'
                    vals_array = np.genfromtxt(vals_file, comments = '#')
                else:
                    raise Exception("Badly defined deriv parameter name %s" % val_names)
            else:
                files_list = glob.glob(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+val_names+'/bin*.txt')
                bin_names = [bin_name.split("/")[-1].replace(".txt", "") for bin_name in files_list]

                bin_vals_dict = {}
                for i, file_name in enumerate(files_list):
                    with open(file_name, 'r') as flx:
                        values = np.loadtxt(flx)
                    bin_vals_dict[bin_names[i]] = values
                
                vals_array = np.array([])
                for bin_name in self.bin_order:
                    vals_array = np.append(vals_array, bin_vals_dict[bin_name])
                
            vals_dict[val_names] = vals_array
        
        return vals_dict
    
    def read_thetas(self, vals_to_read):
        if isinstance(vals_to_read, list):
            vals_to_read = vals_to_read
        elif isinstance(vals_to_read, str):
            vals_to_read = [vals_to_read]
        else:
            raise Exception("Badly defined values to read, needs to be of type string or list")

        theta_dict = {}
        for val_names in vals_to_read:
            files_list = glob.glob(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+val_names+'/theta_bin*.txt')
            bin_names = [bin_name.split("/")[-1].replace(".txt", "").replace("theta_", "") for bin_name in files_list]

            bin_vals_dict = {}
            for i, file_name in enumerate(files_list):
                with open(file_name, 'r') as flx:
                    values = np.loadtxt(flx)
                bin_vals_dict[bin_names[i]] = values
            
            vals_array = np.array([])
            for bin_name in self.bin_order:
                vals_array = np.append(vals_array, bin_vals_dict[bin_name])
            
            theta_dict[val_names] = vals_array

        return theta_dict
    
    def get_params(self, parameter_list):
        """
        Gets parameters from the specified mock run
        """
        parameter_dict = {} # Dictionary of parameters to write to file
        for item in parameter_list:
            header, name = item.split("--")
            param_val = self.read_param_from_txt_file(file_location = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+header+'/values.txt', parameter = name)
            parameter_dict[item] = param_val
        print("Fetched parameters are: %s" % str(parameter_dict))
        param_dict = parameter_dict
        return(parameter_dict)

    def read_theory(self, which_theory = "theory"):
        if which_theory == "theory":
            cov_folder = "theory_data_covariance"
        else:
            cov_folder = "theory_data_covariance_" + which_theory + "_deriv"

        theory_file = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+cov_folder+'/theory.txt'
        theory = np.loadtxt(theory_file, skiprows = 1)
        return theory

    def read_covariance(self, which_cov = "covariance"):
        if which_cov == "covariance":
            cov_folder = "theory_data_covariance"
        else:
            cov_folder = "theory_data_covariance_" + which_cov + "_deriv"

        covariance_file = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+cov_folder+'/covariance.txt'
        covariance = np.loadtxt(covariance_file, skiprows = 1)
        return covariance
    
    def read_inv_covariance(self, which_cov = "covariance"):
        if which_cov == "covariance":
            cov_folder = "theory_data_covariance"
        else:
            cov_folder = "theory_data_covariance_" + which_cov + "_deriv"

        inv_covariance_file = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+cov_folder+'/inv_covariance.txt'
        inv_covariance = np.loadtxt(inv_covariance_file, skiprows = 1)
        return inv_covariance
    
    def read_likelihood(self, like_name):
        like_val = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/likelihoods/values.txt'
        like_val = self.read_param_from_txt_file(file_location = like_val, parameter = like_name)
        return like_val
    
    def read_noisey_data(self):
        data_file = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/theory_data_covariance/noise_mean.txt'
        noisey_data = np.genfromtxt(data_file, comments = '#')
        return noisey_data

def run_kcap_deriv(mock_run, param_to_vary, params_to_fix, vals_to_diff, step_size, stencil_pts, mocks_dir = None, mocks_name = None, cleanup = 2):
    """
    Cleanup == 2 means cleaning everything, cleanup = 1 means only cleaning up the temp deriv folder but keeps the dx values
    """
    kcap_run = kcap_deriv(mock_run = mock_run, 
                          param_to_vary = param_to_vary, 
                          params_to_fix = params_to_fix,
                          vals_to_diff = vals_to_diff,
                          mocks_dir = mocks_dir, 
                          mocks_name = mocks_name)
    check = kcap_run.check_existing_derivs()
    if check is True:
        print("All files found for these parameters, skipping this particular deriv run")
    else:
        print("Not all values found, continuing script...")
        pass
        params = kcap_run.get_params()
        step_size, abs_step_size = kcap_run.write_deriv_values(step_size = step_size, stencil_pts = stencil_pts)
        kcap_run.run_deriv_kcap(mpi_opt = False, cluster = True)
        kcap_run.poll_cluster_finished(stencil_pts = stencil_pts)
        kcap_run.copy_deriv_vals_to_mocks(step_size = step_size, abs_step_size = abs_step_size, stencil_pts = stencil_pts)
        kcap_run.first_deriv(abs_step_size = abs_step_size, stencil_pts = stencil_pts)
        if cleanup == 0:
            pass
        elif cleanup == 1:
            kcap_run.cleanup_deriv_folder()
        elif cleanup == 2:
            kcap_run.cleanup_deriv_folder()
            kcap_run.cleanup_dx(stencil_pts = stencil_pts)

def run_omega_m_deriv(mock_run, params_varied, vals_to_diff, mocks_dir = None, mocks_name = None):
    kcap_run = kcap_deriv(mock_run = mock_run, 
                          param_to_vary = "cosmological_parameters--omega_m", 
                          params_to_fix = [None],
                          vals_to_diff = vals_to_diff,
                          mocks_dir = mocks_dir, 
                          mocks_name = mocks_name)
    check = kcap_run.check_existing_derivs()
    if check is True:
        print("All files found for these parameters, skipping this particular deriv run")
    else:
        print("Not all values found, continuing script...")
        kcap_run.first_omega_m_deriv(params_varied)

def calc_inv_covariance(covariance, which_cov = "eigen"):
    if which_cov == "symmetrised":
        # Averaging the inverse covariance
        inv_covariance = np.linalg.inv(covariance)
        inv_covariance = (inv_covariance + inv_covariance.T)/2
    elif which_cov == "cholesky":
        # Cholesky method for calculating the inverse covariance
        cholesky_decom = np.linalg.cholesky(covariance)
        identity = np.identity(len(covariance))
        y = np.linalg.solve(cholesky_decom, identity)
        inv_covariance = np.linalg.solve(cholesky_decom.T, y)
    elif which_cov == "eigen":
        # Eigenvalue decomposition method of calculating the inverse covariance
        eigenvals, eigenvectors = np.linalg.eig(covariance)
        inv_eigenvals = np.zeros(shape = (270,270))
        for i, val in enumerate(eigenvals):
            if val > 0.:
                inv_eigenvals[i][i] += 1/val
            else:
                pass
        inv_covariance = np.dot(eigenvectors, np.dot(inv_eigenvals, eigenvectors.T))
    elif which_cov == "suppressed":
        for i, cov_slice in enumerate(covariance):
            for j, val in enumerate(cov_slice):
                covariance[i][j] = val * 0.9 ** (abs(i-j))
        inv_covariance = np.inv(covariance)
    else:
        inv_covariance = np.linalg.inv(covariance)
        
    return inv_covariance

def get_values(mock_run, vals_to_read, mocks_dir = None, mocks_name = None, bin_order = None):
    values_method = read_kcap_values(mock_run = mock_run, mocks_dir = mocks_dir, mocks_name = mocks_name, bin_order = bin_order)
    values_read = values_method.read_vals(vals_to_read = vals_to_read)
    return values_read

def get_thetas(mock_run, vals_to_read, mocks_dir = None, mocks_name = None, bin_order = None):
    values_method = read_kcap_values(mock_run = mock_run, mocks_dir = mocks_dir, mocks_name = mocks_name, bin_order = bin_order)
    values_read = values_method.read_thetas(vals_to_read = vals_to_read)
    return values_read

def get_params(mock_run, vals_to_read, mocks_dir = None, mocks_name = None):
    values_method = read_kcap_values(mock_run = mock_run, mocks_dir = mocks_dir, mocks_name = mocks_name)
    values_read = values_method.get_params(parameter_list = vals_to_read)
    return values_read

def get_theory(mock_run, which_cov = "theory", mocks_dir = None, mocks_name = None):
    values_method = read_kcap_values(mock_run = mock_run, mocks_dir = mocks_dir, mocks_name = mocks_name)
    covariance = values_method.read_theory(which_cov = which_cov)
    return covariance

def get_covariance(mock_run, which_cov = "covariance", mocks_dir = None, mocks_name = None):
    values_method = read_kcap_values(mock_run = mock_run, mocks_dir = mocks_dir, mocks_name = mocks_name)
    covariance = values_method.read_covariance(which_cov = which_cov)
    return covariance

def get_inv_covariance(mock_run, which_cov = "covariance", mocks_dir = None, mocks_name = None):
    values_method = read_kcap_values(mock_run = mock_run, mocks_dir = mocks_dir, mocks_name = mocks_name)
    covariance = values_method.read_covariance(which_cov = "covariance")
    inv_covariance = calc_inv_covariance(covariance, which_cov)
    return inv_covariance

def get_noisey_data(mock_run, mocks_dir = None, mocks_name = None):
    values_method = read_kcap_values(mock_run = mock_run, mocks_dir = mocks_dir, mocks_name = mocks_name)
    noisey_data = values_method.read_noisey_data()
    return noisey_data

def get_likelihood(mock_run, like_name = "loglike_like", mocks_dir = None, mocks_name = None):
    values_method = read_kcap_values(mock_run = mock_run, mocks_dir = mocks_dir, mocks_name = mocks_name)
    like_val = values_method.read_likelihood(like_name = like_name)
    return like_val

def get_fiducial_deriv(deriv_params, data_params, fiducial_run = 0, mocks_dir = None, mocks_name = None, bin_order = None):
    for i, deriv_param in enumerate(deriv_params):
        if i == 0:
            deriv_vals_to_get = [data_param + '_' + deriv_param + '_deriv' for data_param in data_params]
            deriv_vector_dict = get_values(mock_run = fiducial_run, vals_to_read = deriv_vals_to_get, mocks_dir = mocks_dir, mocks_name = mocks_name, bin_order = bin_order)
            deriv_vector = np.array([])
            for data_deriv_param in deriv_vals_to_get:
                deriv_vector = np.append(deriv_vector, deriv_vector_dict[data_deriv_param])
            deriv_matrix = np.zeros(shape = (len(deriv_params), len(deriv_vector)))
        else:
            deriv_vals_to_get = [data_param + '_' + deriv_param + '_deriv' for data_param in data_params]
            deriv_vector_dict = get_values(mock_run = fiducial_run, vals_to_read = deriv_vals_to_get, mocks_dir = mocks_dir, mocks_name = mocks_name, bin_order = bin_order)
            deriv_vector = np.array([])
            for data_deriv_param in deriv_vals_to_get:
                deriv_vector = np.append(deriv_vector, deriv_vector_dict[data_deriv_param])
        
        deriv_matrix[i] += deriv_vector
    
    return deriv_matrix

def get_single_data_vector(mock_run, data_params, mocks_dir = None, mocks_name = None, bin_order = None):
    """
    Returns a flattened data vector
    """
    data_vector_dict = get_values(mock_run = mock_run, vals_to_read = data_params, mocks_dir = mocks_dir, mocks_name = mocks_name, bin_order = bin_order)
    data_vector = np.array([])
    for data_param in data_params:
        data_vector = np.append(data_vector, data_vector_dict[data_param])
    
    return data_vector

def get_fiducial_cov_deriv(fiducial_run, deriv_matrix, deriv_params, mocks_dir = None, mocks_name = None):
    cov_tensor_shape = list(deriv_matrix.shape)
    cov_tensor_shape.append(cov_tensor_shape[-1])
    cov_deriv_tensor = np.zeros(shape = cov_tensor_shape)
    for i, deriv_param in enumerate(deriv_params):
        cov_deriv = get_covariance(mock_run = fiducial_run, which_cov = deriv_param, mocks_dir = mocks_dir, mocks_name = mocks_name)
        cov_deriv_tensor[i] += cov_deriv
    
    return cov_deriv_tensor

def get_sim_batch_likelihood(sim_number, mocks_dir = None, mocks_name = None):
    """
    Wrapper function to fetch the gaussian likelihood as calculated by KCAP
    """

    likelihood = np.zeros(sim_number)
    for i in range(sim_number):
        likelihood[i] += get_likelihood(mock_run = i, like_name = "loglike_like", mocks_dir = mocks_dir, mocks_name = mocks_name)
    
    return likelihood 

def get_sim_batch_data_vectors(sim_number, data_vector_length = 270, mocks_dir = None, mocks_name = None, noisey_data = True):
    """
    Fetches the data vector
    """
    sim_data_vector = np.zeros(shape = (sim_number, data_vector_length))

    for i in range(sim_number):
        # Fetch the datavector
        if noisey_data is True:
            data_vector = get_noisey_data(mock_run = i, mocks_dir = mocks_dir, mocks_name = mocks_name)

        sim_data_vector[i] = data_vector

    print("Fetched values!")

    return sim_data_vector

def get_sim_batch_thetas(sim_number, theta_names, mocks_dir = None, mocks_name = None):
    """
    Fetches all of the simulation theta values
    """
    thetas = np.zeros(shape = (sim_number, len(theta_names)))
    for i in range(sim_number):
        theta = get_params(mock_run = i, vals_to_read = theta_names, mocks_dir = mocks_dir, mocks_name = mocks_name)
        theta = np.array(list(theta.values()))
        thetas[i] += theta
    
    return thetas

def cleanup_folders(mock_run_start, num_mock_runs, mocks_dir = None, mocks_name = None,
                   folders_to_keep = ["shear_xi_minus_binned", 
                                      "shear_xi_plus_binned", 
                                      "cosmological_parameters",
                                      "intrinsic_alignment_parameters",
                                      "growth_parameters",
                                      "bias_parameters",
                                      "halo_model_parameters",
                                      "likelihoods",
                                      "theory_data_covariance",
                                      "delta_z_out",
                                      "nofz_shifts",
                                      "nz_source",
                                      "priors"],
                   files_to_remove = ["theory_data_covariance/covariance.txt", "theory_data_covariance/inv_covariance.txt"]):
    print("Checking files found first...")
    clean_method = organise_kcap_output(mock_run_start = mock_run_start, num_mock_runs = num_mock_runs, mocks_dir = mocks_dir, mocks_name = mocks_name, 
                                        folders_to_keep = folders_to_keep, files_to_remove = files_to_remove)
    clean_method.extract_all_runs()
    print("Initiating file cleanup...")
    clean_method.delete_all_unwanted()
    clean_method.delete_all_tgz()
    print("Enjoy that sweet sweet disk space!")

def extract_and_cleanup(mock_run_start, num_mock_runs, mocks_dir = None, mocks_name = None,
                   folders_to_keep = ["shear_xi_minus_binned", 
                                      "shear_xi_plus_binned", 
                                      "cosmological_parameters",
                                      "intrinsic_alignment_parameters",
                                      "growth_parameters",
                                      "bias_parameters",
                                      "halo_model_parameters",
                                      "likelihoods",
                                      "theory_data_covariance",
                                      "delta_z_out",
                                      "nofz_shifts",
                                      "nz_source",
                                      "priors"],
                   files_to_remove = ["theory_data_covariance/covariance.txt", "theory_data_covariance/inv_covariance.txt"]):
    print("Initiating unzip and cleanup afterwards... ")
    clean_method = organise_kcap_output(mock_run_start = mock_run_start, num_mock_runs = num_mock_runs, mocks_dir = mocks_dir, mocks_name = mocks_name, 
                                        folders_to_keep = folders_to_keep, files_to_remove = files_to_remove)
    clean_method.extract_and_delete()
    print("Enjoy that sweet sweet disk space and your extracted files!")
    
if __name__ == "__main__":
    extract_and_cleanup(mock_run_start = 0, num_mock_runs = 2000, mocks_dir = '/share/data1/klin/kcap_out/kids_1000_mocks/trial_33/hypercube_1500_fisher_1000_iterations',
                   mocks_name = 'kids_1000_cosmology_with_nz_shifts_corr')

# For regular deriv calcs -----------------------------------------------------------------------------------------------------

    # run_kcap_deriv(mock_run = 0, 
    #             param_to_vary = "cosmological_parameters--omch2",
    #             params_to_fix = ["cosmological_parameters--sigma_8",
    #                              "cosmological_parameters--n_s", 
    #                              "cosmological_parameters--ombh2",
    #                              "halo_model_parameters--a",
    #                              "cosmological_parameters--h0",
    #                              "intrinsic_alignment_parameters--a",
    #                              "nofz_shifts--bias_1",
    #                              "nofz_shifts--bias_2",
    #                              "nofz_shifts--bias_3",
    #                              "nofz_shifts--bias_4",
    #                              "nofz_shifts--bias_5"],
    #             vals_to_diff = ["theory"],
    #             step_size = 0.01,
    #             stencil_pts = 5,
    #             mocks_dir = '/share/data1/klin/kcap_out/kids_fiducial_data_mocks',
    #             mocks_name = 'kids_1000_cosmology_fiducial',
    #             cleanup = 2
    #             )

    # run_omega_m_deriv(mock_run = 0, 
    #                params_varied = ["cosmological_parameters--omch2"],
    #                vals_to_diff = ["theory"],
    #                mocks_dir = '/home/ruyi/cosmology/kcap_output/kids_mocks',
    #                mocks_name = 'kids_1000_cosmology_noiseless'
    #                )