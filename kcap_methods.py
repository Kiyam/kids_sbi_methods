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
from environs import Env

#TODO - Need to set it so that the stepsize is a relative stepsize, typically of order 1*10^-2 -> 1*10^-5

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
            self.extract_tar(mock_run = mock_run)
            print("Succesfully untarred")
        else:
            raise Exception("Sorry, the requested mock run doesn't exist")
        return mock_run

    def extract_tar(self, mock_run, option = "mocks"):
        """
        Untars file
        """
        if option == "mocks":
            with tarfile.open(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+mock_run+'.tgz', 'r:gz') as kcap_tar:
                kcap_tar.extractall(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+mock_run)
                kcap_tar.close()
                print("Mock run extracted is: %s" % mock_run)
        elif option == "derivs":
            for step in range(4):
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
                elif name == "s_8":
                    name = "s_8_input"
                values_config[header][name] = str(self.param_dict[param])
            else:
                raise Exception("Unknown parameter specified in params_to_fix")

        if self.param_to_vary in self.param_dict:
            middle_val = self.param_dict[self.param_to_vary]
            abs_step_size = np.absolute(middle_val) * step_size
            lower_two_step = middle_val - 2*abs_step_size
            lower_one_step = middle_val - abs_step_size
            up_one_step = middle_val + abs_step_size
            up_two_step = middle_val + 2*abs_step_size
            values_list_file = open(self.kids_deriv_values_list, "w")
            new_param_string = str(lower_two_step) + " " + str(middle_val) + " " + str(up_two_step)

            if self.param_name == "sigma_8":
                name = "sigma_8_input"
            elif self.param_name == "s_8":
                name = "s_8_input"
            else:
                name = self.param_name
            file_text = ["#"+self.param_header+"--"+name+"\n", str(lower_two_step)+"\n"+str(lower_one_step)+"\n"+str(up_one_step)+"\n"+str(up_two_step)]
            values_config[self.param_header][name] = new_param_string
            values_list_file.writelines(file_text)
            values_list_file.close()

        else:
            raise Exception("Badly defined parameter to vary...")

        with open(self.kids_deriv_values_file, 'w') as configfile:
            values_config.write(configfile)
        
        return step_size, abs_step_size

    def run_deriv_kcap(self, mpi_opt, threads):
        self.check_ini_settings(ini_file_to_check = 'deriv_file')
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
            f.write(self.param_to_vary+"_relative_stepsize="+str(step_size) + "\n" + self.param_to_vary+"_absolute_stepsize="+str(abs_step_size))

        for deriv_run in range(4):
            for param in self.vals_to_diff:
                new_subdir_root = param + "_" + self.param_name
                shutil.copytree(self.kids_deriv_dir+'/'+self.kids_deriv_root_name+'_'+str(deriv_run)+'/'+param, 
                                self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+new_subdir_root+step_list[deriv_run])
    
    def check_existing_derivs(self):
        print("Checking if the corresponding derivatives exist...")
        check_count = 0
        for deriv_vals in self.vals_to_diff:
            if "covariance" in deriv_vals:
                if os.path.exists(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'/covariance.txt'):
                    print("Files for %s numerical derivative values wrt to %s found." % (deriv_vals, self.param_name))
                    check_count += 1
                else:
                    print("Missing derivatives for %s wrt to %s." % (deriv_vals, self.param_name))
            else:      
                num_bins = len(glob.glob(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'/bin*.txt'))
                num_found_bins = len(glob.glob(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_name+'_deriv/bin*.txt'))
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

    def cleanup_deriv_folder(self):
        print("Checking all files exist as expected and cleaning up...")
        files_found = self.check_existing_derivs()
        if files_found is True:
            print("Initiating cleanup...")
            shutil.rmtree(self.kids_deriv_dir)
            os.makedirs(self.kids_deriv_dir)
            check_dir = os.listdir(self.kids_deriv_dir)
            
            if len(check_dir) == 0:
                if len(glob.glob(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/*_'+self.param_name+'*dx')) == 0:
                    print("Derivatives temprorary files succesfully cleaned up.")
            else:
                raise Exception("Error during directory cleanup, please manually inspect!")
        else:
            print("Not all files found, exiting cleanup. Please manually inspect!")
            exit()
    
    def cleanup_dx(self):
        for deriv_vals in self.vals_to_diff:
            shutil.rmtree(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_name+'-2dx/')
            shutil.rmtree(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_name+'-1dx/')
            shutil.rmtree(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_name+'+2dx/')
            shutil.rmtree(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_name+'+1dx/')

    def first_deriv(self, abs_step_size):
        """
        Calculates the first derivative using a 5 point stencil
        """
        for deriv_vals in self.vals_to_diff:
            if "covariance" in deriv_vals:
                minus_2dx_file = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_name+'-2dx/covariance.txt'
                minus_1dx_file = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_name+'-1dx/covariance.txt'
                plus_2dx_file = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_name+'+2dx/covariance.txt'
                plus_1dx_file = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_name+'+1dx/covariance.txt'

                with open(minus_2dx_file, 'r') as flx: 
                    minus_2dx_vals = np.loadtxt(flx)
                with open(minus_1dx_file, 'r') as flx: 
                    minus_1dx_vals = np.loadtxt(flx)
                with open(plus_2dx_file, 'r') as flx: 
                    plus_2dx_vals = np.loadtxt(flx)
                with open(plus_1dx_file, 'r') as flx: 
                    plus_1dx_vals = np.loadtxt(flx)
            else:
                minus_2dx_files = glob.glob(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_name+'-2dx/bin*.txt')
                minus_1dx_files = glob.glob(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_name+'-1dx/bin*.txt')
                plus_2dx_files = glob.glob(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_name+'+2dx/bin*.txt')
                plus_1dx_files = glob.glob(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_name+'+1dx/bin*.txt')

                assert len(minus_2dx_files) == len(minus_1dx_files) == len(plus_2dx_files) == len(plus_1dx_files), "Some dx stepsize files missing."
                
                #fetch bin names
                bin_names = [bin_name.split("/")[-1].replace(".txt", "") for bin_name in minus_2dx_files]

                minus_2dx_vals = np.array([])
                for dx_file_name in minus_2dx_files:
                    with open(dx_file_name, 'r') as flx:
                        values = np.loadtxt(flx)
                    minus_2dx_vals = np.append(minus_2dx_vals, values)
                minus_2dx_vals = minus_2dx_vals.reshape((len(bin_names), -1))
                
                minus_1dx_vals = np.array([])
                for dx_file_name in minus_1dx_files:
                    with open(dx_file_name, 'r') as flx:
                        values = np.loadtxt(flx)
                    minus_1dx_vals = np.append(minus_1dx_vals, values)
                minus_1dx_vals = minus_1dx_vals.reshape((len(bin_names), -1))
                
                plus_2dx_vals = np.array([])
                for dx_file_name in plus_2dx_files:
                    with open(dx_file_name, 'r') as flx:
                        values = np.loadtxt(flx)
                    plus_2dx_vals = np.append(plus_2dx_vals, values)
                plus_2dx_vals = plus_2dx_vals.reshape((len(bin_names), -1))
                
                plus_1dx_vals = np.array([])
                for dx_file_name in plus_1dx_files:
                    with open(dx_file_name, 'r') as flx:
                        values = np.loadtxt(flx)
                    plus_1dx_vals = np.append(plus_1dx_vals, values)
                plus_1dx_vals = plus_1dx_vals.reshape((len(bin_names), -1))

            print("All values needed for %s derivatives wrt to %s found, calculating and saving to file..." % (deriv_vals, self.param_name))

            first_deriv_vals = ((1/12)*minus_2dx_vals - (2/3)*minus_1dx_vals + (2/3)*plus_1dx_vals - (1/12)*plus_2dx_vals)/abs_step_size

            deriv_dir_path = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+self.param_name+"_deriv/"
            if not os.path.exists(os.path.dirname(deriv_dir_path)):
                try:
                    os.makedirs(os.path.dirname(deriv_dir_path))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            if "covariance" in deriv_vals:
                    deriv_file = deriv_dir_path+"covariance.txt"
                    np.savetxt(deriv_file, first_deriv_vals, newline="\n", header="covariance")
            else:
                for i, vals in enumerate(first_deriv_vals):
                    deriv_file = deriv_dir_path+bin_names[i]+".txt"
                    np.savetxt(deriv_file, vals, newline="\n", header=bin_names[i])
        print("Derivatives saved succesfully")

    def first_omega_m_deriv(self):
        if "omch2" in self.param_to_vary:
            h = self.read_param_from_txt_file(file_location = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/cosmological_parameters/values.txt', parameter = 'h0')
            for deriv_vals in self.vals_to_diff:
                if "covariance" in deriv_vals:
                    omch_deriv_file = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+'omch2_deriv/covariance.txt'
                    with open(omch_deriv_file, 'r') as flx:
                        omch_deriv = np.loadtxt(flx)
                else:
                    omch_deriv_files = glob.glob(self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+'omch2_deriv/bin*.txt')
                    bin_names = [bin_name.split("/")[-1].replace(".txt", "") for bin_name in omch_deriv_files]
                    omch_deriv = np.array([])
                    for dx_file_name in omch_deriv_files:
                        with open(dx_file_name, 'r') as flx:
                            values = np.loadtxt(flx)
                        omch_deriv = np.append(omch_deriv, values)
                    omch_deriv = omch_deriv.reshape((len(bin_names), -1))

                omega_m_deriv_vals = omch_deriv * (h**2)
                
                print("All values needed for %s derivatives wrt to omega_m calculated, saving to file..." % (deriv_vals))

                deriv_dir_path = self.kids_mocks_dir+'/'+self.kids_mocks_root_name+'_'+self.mock_run+'/'+deriv_vals+'_'+'omega_m_deriv/'
                if not os.path.exists(os.path.dirname(deriv_dir_path)):
                    try:
                        os.makedirs(os.path.dirname(deriv_dir_path))
                    except OSError as exc: # Guard against race condition
                        if exc.errno != errno.EEXIST:
                            raise

                if "covariance" in deriv_vals:
                    deriv_file = deriv_dir_path+"covariance.txt"
                    np.savetxt(deriv_file, omega_m_deriv_vals, newline="\n", header="covariance")
                else:
                    for i, vals in enumerate(omega_m_deriv_vals):
                        deriv_file = deriv_dir_path+bin_names[i]+".txt"
                        np.savetxt(deriv_file, vals, newline="\n", header=bin_names[i])
            
            print("Derivatives saved succesfully")
        else:
            print("Not running omega_m_deriv as omch2 has not been varied in this deriv run")
            pass

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
            # self.bin_order = ['bin_1_1', 
            #                   'bin_2_1', 'bin_2_2', 
            #                   'bin_3_1', 'bin_3_2', 'bin_3_3', 
            #                   'bin_4_1', 'bin_4_2', 'bin_4_3', 'bin_4_4', 
            #                   'bin_5_1', 'bin_5_2', 'bin_5_3', 'bin_5_4', 'bin_5_5']
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

def run_kcap_deriv(mock_run, param_to_vary, params_to_fix, vals_to_diff, step_size, mocks_dir = None, mocks_name = None, cleanup = 2):
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
        step_size, abs_step_size = kcap_run.write_deriv_values(step_size = step_size)
        kcap_run.run_deriv_kcap(mpi_opt = True, threads = 4)
        kcap_run.copy_deriv_vals_to_mocks(step_size = step_size, abs_step_size = abs_step_size)
        kcap_run.first_deriv(abs_step_size = abs_step_size)
        kcap_run.first_omega_m_deriv()
        if cleanup == 0:
            pass
        elif cleanup == 1:
            kcap_run.cleanup_deriv_folder()
        elif cleanup == 2:
            kcap_run.cleanup_deriv_folder()
            kcap_run.cleanup_dx()

def get_values(mock_run, vals_to_read, mocks_dir = None, mocks_name = None, bin_order = None):
    values_method = read_kcap_values(mock_run = mock_run, mocks_dir = mocks_dir, mocks_name = mocks_name, bin_order = bin_order)
    values_read = values_method.read_vals(vals_to_read = vals_to_read)
    return values_read

def get_theta(mock_run, vals_to_read, mocks_dir = None, mocks_name = None, bin_order = None):
    values_method = read_kcap_values(mock_run = mock_run, mocks_dir = mocks_dir, mocks_name = mocks_name, bin_order = bin_order)
    values_read = values_method.read_thetas(vals_to_read = vals_to_read)
    return values_read

def get_params(mock_run, vals_to_read, mocks_dir = None, mocks_name = None):
    values_method = read_kcap_values(mock_run = mock_run, mocks_dir = mocks_dir, mocks_name = mocks_name)
    values_read = values_method.get_params(parameter_list = vals_to_read)
    return values_read

def get_covariance(mock_run, which_cov = "covariance", mocks_dir = None, mocks_name = None):
    values_method = read_kcap_values(mock_run = mock_run, mocks_dir = mocks_dir, mocks_name = mocks_name)
    covariance = values_method.read_covariance(which_cov = which_cov)
    return covariance

def get_inv_covariance(mock_run, which_cov = "covariance", mocks_dir = None, mocks_name = None):
    values_method = read_kcap_values(mock_run = mock_run, mocks_dir = mocks_dir, mocks_name = mocks_name)
    inv_covariance = values_method.read_inv_covariance(which_cov = which_cov)
    return inv_covariance

def get_noisey_data(mock_run, mocks_dir = None, mocks_name = None):
    values_method = read_kcap_values(mock_run = mock_run, mocks_dir = mocks_dir, mocks_name = mocks_name)
    noisey_data = values_method.read_noisey_data()
    return noisey_data

def get_likelihood(mock_run, like_name = "loglike_like", mocks_dir = None, mocks_name = None):
    values_method = read_kcap_values(mock_run = mock_run, mocks_dir = mocks_dir, mocks_name = mocks_name)
    like_val = values_method.read_likelihood(like_name = like_name)
    return like_val

if __name__ == "__main__":
    run_kcap_deriv(mock_run = 0, 
                   param_to_vary = "cosmological_parameters--omch2", 
                   params_to_fix = ["cosmological_parameters--sigma_8", "intrinsic_alignment_parameters--a"],
                   vals_to_diff = ["shear_xi_minus_binned", "shear_xi_plus_binned", "theory_data_covariance"],
                   step_size = 0.01)
    run_kcap_deriv(mock_run = 0, 
                   param_to_vary = "cosmological_parameters--omch2", 
                   params_to_fix = ["cosmological_parameters--sigma_8"],
                   vals_to_diff = ["shear_xi_minus_binned", "shear_xi_plus_binned"],
                   step_size = 0.01,
                   mocks_dir = '/home/ruyi/cosmology/kcap_output/kids_mocks',
                   mocks_name = 'kids_1000_cosmology_fiducial',
                   cleanup = 2
                   )
    # run_kcap_deriv(mock_run = 0, 
    #                param_to_vary = "cosmological_parameters--sigma_8", 
    #                params_to_fix = ["cosmological_parameters--omch2"],
    #                vals_to_diff = ["shear_xi_minus_binned", "shear_xi_plus_binned"],
    #                step_size = 0.01,
    #                mocks_dir = '/home/ruyi/cosmology/kcap_output/kids_mocks',
    #                mocks_name = 'kids_1000_cosmology_fiducial',
    #                cleanup = 2
    #                )
    # run_kcap_deriv(mock_run = 0, 
    #                param_to_vary = "cosmological_parameters--s_8", 
    #                params_to_fix = ["cosmological_parameters--omch2"],
    #                vals_to_diff = ["shear_xi_minus_binned", "shear_xi_plus_binned", "theory_data_covariance"],
    #                step_size = 0.01,
    #                mocks_dir = '/home/ruyi/cosmology/kcap_output/kids_mocks',
    #                mocks_name = 'kids_1000_cosmology_fiducial',
    #                cleanup = 2
    #                )

    # temp_vals = get_values(mock_run = 3000, vals_to_read = ["shear_xi_plus_binned", "shear_xi_minus_binned"], mocks_dir = "/home/ruyi_wsl/kcap_output/kids_1000_mocks_trial_2", mocks_name = "kids_1000_cosmology")
    # print(len(temp_vals['shear_xi_plus_binned']))
    # print(len(temp_vals['shear_xi_minus_binned']))
    # covariance, inv_covariance = get_covariance(mock_run = 0)
    # print(len(covariance))
    # print(len(covariance[0]))