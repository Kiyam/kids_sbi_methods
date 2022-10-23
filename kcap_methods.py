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
import pyDOE as pydoe
from scipy.stats.distributions import norm
from pathlib import Path

class read_kcap_values:
    def __init__(self, mock_run = None, mocks_dir = None, mocks_name = None, bin_order = None):
        # mocks_dir settings
        self.mocks_dir = mocks_dir
        self.mocks_name = mocks_name 
        self.mock_run = self.check_mock_run_exists(mock_run)
        
        if self.mock_run is None:
            self.mock_path = '{0}/{1}'.format(self.mocks_dir, self.mocks_name)
        else:
            self.mock_path = '{0}/{1}_{2}'.format(self.mocks_dir, self.mocks_name, self.mock_run)
        self.content = self.fetch_tar_content(self.mock_run)

        if bin_order == None:
            self.bin_order = ['bin_1_1', 'bin_2_1', 'bin_3_1', 'bin_4_1', 'bin_5_1',
                              'bin_2_2', 'bin_3_2', 'bin_4_2', 'bin_5_2',
                              'bin_3_3', 'bin_4_3', 'bin_5_3',
                              'bin_4_4', 'bin_5_4',
                              'bin_5_5']
        else:
            self.bin_order = bin_order

    def check_mock_run_exists(self, mock_run):
        """
        Checks if the requested mock_run file exists, and if not will check for a .tgz file of the same file name and untar as necessary
        """
        if mock_run is None:
            if os.path.exists(self.mocks_dir+'/'+self.mocks_name+'.tgz'):
                return None
            else:
                raise Exception("Sorry, the requested mock run doesn't exist")
        else:
            if os.path.exists(self.mocks_dir+'/'+self.mocks_name+'_'+str(mock_run)+'.tgz'):
                return str(mock_run)
            else:
                raise Exception("Sorry, the requested mock run, %s, doesn't exist" % mock_run)

    def fetch_tar_content(self, mock_run):
        """
        Untars file
        """
        try:
            content = tarfile.open(self.mock_path+'.tgz', 'r:gz')
        except:
            raise Except("Badly formed tarfile")
        
        return content
    
    def read_vals(self, vals_to_read):
        if isinstance(vals_to_read, list):
            vals_to_read = vals_to_read
        elif isinstance(vals_to_read, str):
            vals_to_read = [vals_to_read]
        else:
            raise Exception("Badly defined values to read, needs to be of type string or list")

        vals_dict = {}
        for val_names in vals_to_read:
            val_folder, val_name = val_names.rsplit("--", 1)
            try:
                if 'bin' in val_name:
                    files_list = [bin_file for bin_file in self.content.getnames() if val_folder+'/'+val_name in bin_file]
                    bin_names = [bin_name.split("/")[-1].replace(".txt", "") for bin_name in files_list]
                    
                    bin_vals_dict = {}
                    for i, file_name in enumerate(files_list):
                        bin_vals_dict[bin_names[i]] = np.loadtxt(self.content.extractfile(file_name))
                    
                    vals_array = np.array([])
                    for bin_name in self.bin_order:
                        vals_array = np.append(vals_array, bin_vals_dict[bin_name])
                else:
                    vals_array = np.loadtxt(self.content.extractfile(self.mock_path+'/'+val_folder+'/'+val_name+'.txt'))      
            except:
                raise Exception("Badly defined parameter name %s" % val_names)               
                
            vals_dict[val_names] = vals_array
        
        return vals_dict
    
    def read_params(self, parameter_list):
        """
        Gets parameters from the specified mock run
        """
        parameter_dict = {} # Dictionary of parameters
        for item in parameter_list:
            header, name = item.split("--")
            for line in self.content.extractfile(self.mock_path+'/'+header+'/values.txt'):
                key, value = [word.strip() for word in line.decode('utf-8').split("=")]
                if key == name:
                    param_val = float(value)
            parameter_dict[item] = param_val
        print("Fetched parameters are: %s" % str(parameter_dict))
        param_dict = parameter_dict
        return(parameter_dict)
    
    def close_tar(self):
        """
        Cleanly closes tarfile after reading
        """
        self.content.close()
        
    def write_to_tar(self, to_write, to_write_headers, file_loc):
        """
        Adds a list of files to the tar
        """
        self.content.extractall()
        self.content.close()
        
        assert isinstance(to_write, list), "to_write needs to be a list"
        assert isinstance(to_write_headers, list), "to_write_headers needs to be a list"
        assert isinstance(file_loc, list), "file_loc needs to be a list"
        assert len(to_write) == len(file_loc) and len(to_write) == len(file_loc), "Amount of data to write doesn't match either the number of file locations or headers provided"
        for i in range(len(to_write)):
            file_path = self.mock_path+'/'+file_loc[i]
            if file_path[-1] == "/": #make sure the parent directory exists
                Path(file_path).mkdir(parents=True, exist_ok=True)
            else:
                Path("/".join(file_path.split("/")[:-1])).mkdir(parents=True, exist_ok=True) 
            np.savetxt(file_path, to_write[i], header = to_write_headers[i])
        
        new_files_list = glob.glob(self.mock_path+'/*/*')
        content = tarfile.open(self.mock_path+'.tgz', 'w:gz')
        for file in new_files_list:
            content.add(file)
        content.close()
        
        self.content = tarfile.open(self.mock_path+'.tgz', 'r:gz')
    
    def del_from_tar(self, files_to_delete):
        """
        Deletes a specific file from within the tar
        """
        self.content.extractall()
        self.content.close()
        
        assert isinstance(files_to_delete, list), "files_to_delete needs to be a list"
        
        for file in files_to_delete:
            os.remove(self.mock_path+'/'+file)
        
        new_files_list = glob.glob(self.mock_path+'/*/*')
        content = tarfile.open(self.mock_path+'.tgz', 'w:gz')
        for file in new_files_list:
            content.add(file)
        content.close()
        
        self.content = tarfile.open(self.mock_path+'.tgz', 'r:gz')
    
    def delete_mock_tgz(self, mock_run):
        tgz_to_delete = self.mocks_dir+'/'+self.mocks_name+'_'+str(mock_run)+'.tgz'
        os.remove(tgz_to_delete)
        print("Removed %s" %str(tgz_to_delete))
        
    def delete_all_tgz(self):
        all_tgz = glob.glob(self.mocks_dir+'/'+self.mocks_name+'*.tgz')
        for tgz_file in all_tgz:
            os.remove(tgz_file)
            print("Removed %s" %str(tgz_file))
        print("All .tgz files removed!")    

class kcap_deriv(read_kcap_values):
    def __init__(self, mock_run, param_to_vary, params_to_fix, vals_to_diff, stencil_pts = 5, step_size = 0.01, 
                 mocks_dir = None, mocks_name = None, deriv_dir = None, deriv_name = None, 
                 deriv_ini_file = None, deriv_values_file = None, deriv_values_list_file = None, sbatch_file = None):
        """
        Gets variable from .env file
        """
        self.sbatch_file = sbatch_file
        self.mocks_dir = mocks_dir
        self.mocks_name = mocks_name
        self.deriv_dir = deriv_dir
        self.deriv_name = deriv_name
        self.deriv_ini_file = deriv_ini_file
        self.deriv_values_file = deriv_values_file
        self.deriv_values_list = deriv_values_list_file
        self.mock_run = self.check_mock_run_exists(mock_run)
        
        if self.mock_run is None:
            self.mock_path = '{0}/{1}'.format(self.mocks_dir, self.mocks_name)
        else:
            self.mock_path = '{0}/{1}_{2}'.format(self.mocks_dir, self.mocks_name, self.mock_run)
            
        self.content = self.fetch_tar_content(self.mock_run)
        self.stencil_pts = stencil_pts
        self.step_size = step_size
        
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
        # if "cosmological_parameters--sigma_8" in self.parameter_list:
        #     self.parameter_list[self.parameter_list.index('cosmological_parameters--sigma_8')] = "cosmological_parameters--sigma_8_input"

        self.param_header, self.param_name = param_to_vary.split("--") 
        self.param_dict = self.read_params(self.parameter_list)

    def check_deriv_ini_settings(self):
        """
        Modifies the deriv_values_list.ini file
        """
        values_config = cfg.ConfigParser()
        
        to_change = 0
        values_config.read(self.deriv_ini_file)
        if values_config['DEFAULT']['RESULTS_PATH'] == self.deriv_dir:
            pass
        else:
            values_config['DEFAULT']['RESULTS_PATH'] = self.deriv_dir
            to_change += 1
            
        if values_config['DEFAULT']['RESULTS_NAME'] == self.deriv_name:
            pass
        else:
            values_config['DEFAULT']['RESULTS_NAME'] = self.deriv_name
            to_change += 1

        if to_change > 0:
            print("Setting a few deriv pipeline ini file values...")
            with open(self.deriv_ini_file, 'w') as configfile:
                values_config.write(configfile)

    def write_deriv_ini_values(self):
        """
        Modifies the deriv_values_list.ini file
        """
        values_config = cfg.ConfigParser()
        values_config.read(self.deriv_values_file)

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
                self.abs_step_size = self.step_size
            else:
                self.abs_step_size = np.absolute(middle_val) * self.step_size
            vals_array = middle_val + dx_array*self.abs_step_size

            new_param_string = str(vals_array[0]) + " " + str(middle_val) + " " + str(vals_array[-1])       
            values_config[self.param_header][name] = new_param_string
            file_text = ["#"+self.param_header+"--"+name+"\n"]
            for val in vals_array:
                file_text.append(str(val) + "\n")

            values_list_file = open(self.deriv_values_list, "w")
            values_list_file.writelines(file_text)
            values_list_file.close()

        else:
            raise Exception("Badly defined parameter to vary...")

        with open(self.deriv_values_file, 'w') as configfile:
            values_config.write(configfile)

    def run_deriv_kcap(self, mpi_opt = False, cluster = True):
        self.check_ini_settings(ini_file_to_check = 'deriv_file')
        if mpi_opt == True:
            subprocess.run(["mpirun", "-n" , str(stencil_points-1), "--use-hwthread-cpus", "cosmosis", "--mpi", self.deriv_ini_file])
        elif cluster == True:
            subprocess.run(["sbatch", self.sbatch_file])
        elif mpi_opt == False:
            subprocess.run(["cosmosis", self.deriv_ini_file])
        else: 
            raise Exception("Failed to initialise cosmosis pipeline for derivatives")
    
    def poll_cluster_finished(self, stencil_pts = 5):
        start_time = time.time()
        elapsed = time.time() - start_time
        finished = False
        while elapsed <= 86400. and finished != True:
            if len(glob.glob(self.deriv_dir+'/'+self.deriv_name+'_*.tgz')) < stencil_pts - 1:
                time.sleep(15)
            elif len(glob.glob(self.deriv_dir+'/'+self.deriv_name+'_*.tgz')) == stencil_pts - 1:
                finished = True
        print("Waiting to ensure all IO operations are finished")
        time.sleep(30)

    def fetch_dx_values(self):
        dx_list = []
        num_deriv_pts = self.stencil_pts - 1
        if len(glob.glob(self.deriv_dir+'/'+self.deriv_name+'_*.tgz')) == num_deriv_pts:
            for deriv_run in range(stencil_pts - 1):
                deriv_tar = read_kcap_values(mock_run = deriv_run, mocks_dir = self.deriv_dir, mocks_name = self.deriv_name)
                vals_dict = deriv_tar.read_vals(vals_to_read = self.vals_to_diff)
                dx_list.append(vals_dict)
        else:
            raise Exception("Sorry, you seem to be missing the derivatives, try running KCAP for the requested derivative steps first")
        
        dx_dict = {}
        for val_to_diff in self.vals_to_diff:
            val_array = np.array([])
            for i in range(num_deriv_pts):
                dx_dict[dx_dict] = np.append(val_array, dx_list[i][val_to_diff])
            val_array = val_array.reshape(num_deriv_pts, -1)
            dx_dict[val_to_diff] = val_array
        
        return dx_dict

    def calc_deriv_values(self):
        """
        Calculates the first derivative using a 5 point stencil
        """
        
        if self.stencil_pts == 3:
            stencil_coeffs = np.array([-1/2, 1/2])
        elif self.stencil_pts == 5:
            stencil_coeffs = np.array([1/12, -2/3, 2/3, -1/12])
        elif self.stencil_pts == 7:
            stencil_coeffs = np.array([-1/60, 3/20, -3/4, 3/4, -3/20, 1/60])
        elif self.stencil_pts == 9:
            stencil_coeffs = np.array([1/280, -4/105, 1/5, -4/5, 4/5, -1/5, 4/105, -1/280])
        else:
            raise Exception("Invalid stencil number inputted")           
        
        dx_dict = self.fetch_dx_values()
        print("All values needed for %s derivatives wrt to %s found, now calculating..." % (deriv_vals, self.param_to_vary))
        deriv_dict = {}
        for val_to_diff in self.vals_to_diff:
            first_deriv_vals = np.dot(stencil_coeffs, dx_dict[val_to_diff])/self.abs_step_size
            deriv_dict[val_to_diff] = first_deriv_vals
        print("Derivatives calculated succesfully")
        
        return deriv_dict
        
    def save_deriv_values(self, deriv_dict):
        to_write = []
        to_write_headers = []
        file_locs = []
        
        for val_to_diff in self.vals_to_diff:
            val_to_diff_head, val_to_diff_name = val_to_diff.split(sep = "--")
            if "bin" in val_to_diff: # This option saves the files that should be binned into their respective bin folders to maintain folder/file structure
                deriv_vals = deriv_dict[val_to_diff].reshape((len(self.bin_order), -1))
                for i, vals in enumerate(deriv_vals):
                    to_write.append(vals)
                    to_write_headers.append('deriv values for {0} wrt {1} with relative stepsize of {3} and absolute stepsize of {4}'.format(val_to_diff, self.param_to_vary, self.step_size, self.abs_step_size))
                    file_locs.append(self.mock_path+'/'+val_to_diff_head+'_'+self.param_to_vary+'_deriv/'+self.bin_order[i]+'.txt')
            else:
                to_write.append(deriv_dict[val_to_diff])
                to_write_headers.append('deriv values for {0} wrt {1} with relative stepsize of {3} and absolute stepsize of {4}'.format(val_to_diff, self.param_to_vary, self.step_size, self.abs_step_size))
                file_locs.append(self.mock_path+'/'+val_to_diff_head+'_'+self.param_to_vary+'_deriv/'+val_to_diff_name+'.txt')
        
        self.write_to_tar(to_write, to_write_headers, file_loc)
            
    def check_existing_derivs(self):
        print("Checking if the corresponding derivatives exist...")
        for val_to_diff in self.vals_to_diff:
            val_to_diff_head, val_to_diff_name = val_to_diff.split(sep = "--")
            if 'bin' in val_to_diff_name:
                num_base = len(self.bin_order)
                num_found = len([bin_file for bin_file in self.content.getnames() if val_to_diff_head+'_'+self.param_to_vary+'_deriv/'+vall_to_diff_name in bin_file])
            else:
                num_base = len([files for files in self.content.getnames() if val_to_diff_head+'/'+vall_to_diff_name in files])
                num_found = len([files for files in self.content.getnames() if val_to_diff_head+'_'+self.param_to_vary+'_deriv/'+vall_to_diff_name in files])
            if num_base == num_found:
                print("Files for %s numerical derivative values wrt to %s found." % (val_to_diff_head, self.param_to_vary))
                pass
            else:
                raise Exception("Missing derivatives for %s wrt to %s." % (val_to_diff_head, self.param_to_vary))
        
        print("All wanted numerical derivative values found!")
        return True

    def cleanup_deriv_folder(self):
        print("Checking all files exist as expected and cleaning up...")
        files_found = self.check_existing_derivs()
        if files_found is True:
            print("Initiating cleanup...")
            shutil.rmtree(self.deriv_dir)
            os.makedirs(self.deriv_dir)
            check_dir = os.listdir(self.deriv_dir)
            
            if len(check_dir) == 0:
                print("Derivatives temprorary files succesfully cleaned up.")
            else:
                raise Exception("Error during directory cleanup, please manually inspect!")
        else:
            raise Exception("Not all files found, exiting cleanup. Please manually inspect!")     

    def omega_m_deriv(self):       
        h = self.read_param_from_txt_file(file_location = self.mocks_dir+'/'+self.mocks_name+'_'+self.mock_run+'/cosmological_parameters/values.txt', parameter = 'h0')
        
        omch2_deriv_names = []
        for val_to_diff in self.vals_to_diff: # make a list of the names of the variables to fetch
            val_to_diff_head, val_to_diff_name = val_to_diff.split(sep = "--")
            omch2_deriv_names.append(self.mock_path+'/'+val_to_diff_head+'_'+self.param_to_vary+'_deriv--'+val_to_diff_name)
            
        omch_deriv = self.read_vals(vals_to_read = omch2_deriv_names)
        to_write = []
        to_write_headers = []
        file_locs = []
        
        for i, val_to_diff in enumerate(self.vals_to_diff):
            val_to_diff_head, val_to_diff_name = val_to_diff.split(sep = "--")
            omega_m_deriv_vals = omch_deriv[omch2_deriv_names[i]] * (h**2)
            
            if "bin" in val_to_diff: # This option saves the files that should be binned into their respective bin folders to maintain folder/file structure
                deriv_vals = omega_m_deriv_vals.reshape((len(self.bin_order), -1))
                for i, vals in enumerate(deriv_vals):
                    to_write.append(vals)
                    to_write_headers.append('deriv values for {0} wrt {1} derived from omch2 deriv'.format(val_to_diff, self.param_to_vary))
                    file_locs.append(self.mock_path+'/'+val_to_diff_head+'_cosmological_parameters--omega_m_deriv/'+self.bin_order[i]+'.txt')
            else:
                to_write.append(omega_m_deriv_vals)
                to_write_headers.append('deriv values for {0} wrt {1} derived from omch2 deriv'.format(val_to_diff, self.param_to_vary))
                file_locs.append(self.mock_path+'/'+val_to_diff_head+'_cosmological_parameters--omega_m_deriv/'+val_to_diff_name+'.txt')
                
            print("All values needed for %s derivatives wrt to omega_m calculated, saving to file..." % (deriv_vals))
            self.write_to_tar(to_write, to_write_headers, file_loc)
            print("Derivatives saved succesfully")

class kcap_delfi(read_kcap_values):
    def __init__(self, params_to_vary, params_to_read, data_name, data_vec_length,
                 mocks_dir = None, 
                 mocks_name = None, 
                 mocks_ini_file = None, 
                 mocks_values_file = None,
                 save_folder = None,
                 nz_indices = None,
                 nz_cov = None,
                 verbose = False,
                 comm = None,
                 slurm_file = None,
                 is_binned = False):
        """
        Gets variable from .env file
        """

        env = Env()
        env.read_env()

        if slurm_file:
            self.is_cluster = True
            self.slurm_file = slurm_file
        else:
            self.is_cluster = False
        
        # mocks_dir settings
        if mocks_dir == None:
            self.mocks_dir = env.str('kids_mocks_dir')
        else:
            self.mocks_dir = mocks_dir

        # mocks_name settings
        if mocks_name == None:
            self.mocks_name = env.str('kids_mocks_root_name')
        else:
            self.mocks_name = mocks_name
        
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
            raise Exception("param_to_vary variable must be a list")
        
        if isinstance(params_to_read, list):
            self.params_to_read = params_to_read
        else:
            raise Exception("params_to_read variable must be a list")

        self.save_folder = save_folder
        self.nz_indices = nz_indices
        self.nz_cov = nz_cov
        self.data_name = data_name
        self.data_vec_length = data_vec_length
        self.mock_run_start = 0 #For the active learning method the mock runs will aways start from number 0
        self.verbose = verbose
        self.folders_to_keep = folders_to_keep
        self.files_to_remove = files_to_remove
        self.is_binned = is_binned
        
        if comm:
            self.comm = comm
        
        print("Simulator succesfully initialized")
    
    def run_delfi_kcap(self, rank = None):
        if self.is_cluster == True:
            output = subprocess.run(["sbatch", self.slurm_file], stdout = subprocess.PIPE, text=True)
        else:
            pipeline_file = self.kids_pipeline_ini_file + '_' + str(rank) + '.ini'
            output = subprocess.run(["cosmosis", pipeline_file])
        
        text_out = output.stdout
        jobid = text_out.split(" ")[-1].rstrip()
        print("------------------------")
        print("Submitted slurm job with jobid: "+jobid)
        print("------------------------")
        return jobid
        
    def write_delfi_kcap_values(self, theta, rank = None):
        """
        Modifies the values_list.ini file
        """
        if theta.ndim > 1:
            assert theta.shape[1] == len(self.params_to_vary), "Different number of parameters to vary fed vs. defined to vary."
        else:
            assert len(theta) == len(self.params_to_vary), "Different number of parameters to vary fed vs. defined to vary."
        values_config = cfg.ConfigParser()
        if rank:
            values_list_file = self.kids_pipeline_values_file + '_' + str(rank) + '.ini'
        else:
            values_list_file = self.kids_pipeline_values_file + '.ini'
        
        # The NZ mixing stuff
        if self.nz_indices:
            inv_L = np.linalg.inv(np.linalg.cholesky(self.nz_cov)) 
            if theta.ndim > 1:
                nz_theta = theta[:,min(self.nz_indices):max(self.nz_indices)+1]
                nz_theta = np.dot(inv_L, nz_theta.T).T
                theta[:,min(self.nz_indices):max(self.nz_indices)+1] = nz_theta 
            else:
                nz_theta = theta[min(self.nz_indices):max(self.nz_indices)+1]
                nz_theta = np.dot(inv_L, nz_theta)
                theta[min(self.nz_indices):max(self.nz_indices)+1] = nz_theta 

        #Just doing some parameter renaming
        params_to_vary = self.params_to_vary
        for i, val in enumerate(params_to_vary):
            if val == "cosmological_parameters--sigma_8":
                params_to_vary[i] = "cosmological_parameters--sigma_8_input"
            elif val == "cosmological_parameters--s_8":
                params_to_vary[i] = "cosmological_parameters--s_8_input"

        #Write the values list to file
        header = " ".join(params_to_vary)
        if theta.ndim > 1:
            np.savetxt(fname = values_list_file, X = theta, header=header)
        else:
            np.savetxt(fname = values_list_file, X = np.array([theta]), header=header)

        print("Succesfully set values to run simulations on")
    
    def poll_cluster_finished(self, jobid):
        start_time = time.time()
        elapsed = time.time() - start_time
        finished = False
        while elapsed <= 1200000. and finished != True:
            try: 
                subprocess.check_output(["squeue", "-j", jobid])
                time.sleep(30)
            except:
                print("Simulations finished!")
                finished = True
        print("Waiting to ensure all IO operations are finished")
        time.sleep(40)
    
    def save_sims(self):
        if self.save_folder:
            which_population = len(glob.glob(self.save_folder + '/*/'))
            shutil.copytree(self.mocks_dir, self.save_folder  + '/population_' + str(which_population))

    def clean_mocks_folder(self):
        shutil.rmtree(self.mocks_dir)
        os.makedirs(self.mocks_dir)

    def simulate(self, theta):
        if theta.ndim > 1:
            self.num_mock_runs = len(theta)
        else:
            self.num_mock_runs = 1
        print("Writing values to file")
        self.write_delfi_kcap_values(theta)
        print("Running KCAP")
        jobid = self.run_delfi_kcap()
        time.sleep(30)
        self.poll_cluster_finished(jobid)
        self.save_sims()
        
        sim_data_vector = np.zeros((self.num_mock_runs, self.data_vec_length))
        thetas = np.zeros((self.num_mock_runs, len(self.params_to_read)))
        for i in range(self.num_mock_runs):
            self.mock_run = str(i)
            try:
                values_read = self.read_vals(vals_to_read = self.data_name)
                data_vector = np.array(list(values_read.values()))
                theta = self.get_params(parameter_list = self.params_to_read)
                theta = np.array(list(theta.values()))

                sim_data_vector[i] = data_vector
                thetas[i] = theta
            except:
                print("Mock run %s doesn't exist, skipping this datavector" % (i))

        assert len(sim_data_vector) == len(thetas), "Mismatch between number of fetched simulation data vectors: %s and parameter sets: %s" %(len(sim_data_vector), len(thetas))
        self.clean_mocks_folder()

        return sim_data_vector, thetas

class kcap_delfi_proposal():
    def __init__(self, n_initial, lower, upper, transformation = None, delta_z_indices = None, delta_z_cov = None, factor_of_safety = 5, iterations = 1000):
        # The lower and upper bounds for this are for a truncated Gaussian, so it should be different to the PyDELFI prior
        assert len(lower) == len(upper)
        num_samples = n_initial * factor_of_safety
        points = pydoe.lhs(len(lower), samples = num_samples, criterion = 'cm', iterations = iterations)
        for i in range(len(lower)):
            if delta_z_indices is not None and i in delta_z_indices: #Does a truncated Gaussian
                p = 0
                while p == 0:
                    x = norm(loc = 0, scale = 1).ppf(points[:,i])
                    p = self.gaussian(lower[i], upper[i], x)
                points[:,i] = x
            else:
                val_range = upper[i] - lower[i]
                points[:, i] *= val_range
                points[:, i] += lower[i]
        
        if transformation is not None:
            points = transformation(points)
        
        if delta_z_cov is not None:
            uncorr_nz = points[:,min(delta_z_indices):max(delta_z_indices)+1]                
            L = np.linalg.cholesky(delta_z_cov)
            corr_nz = np.inner(L, uncorr_nz).T
            points[:,min(delta_z_indices):max(delta_z_indices)+1] = corr_nz

        self.sample_points = iter(points)
    
    def gaussian(self, g_lower, g_upper, x):
        inrange = np.prod(x > g_lower)*np.prod(x < g_upper)
        return inrange*np.prod(g_upper-g_lower)
    
    def draw(self):
        return next(self.sample_points)

def run_kcap_deriv(mock_run, param_to_vary, params_to_fix, vals_to_diff, step_size, stencil_pts, 
                   mocks_dir = None, mocks_name = None, cleanup = 2,
                   deriv_dir = None, deriv_name = None, deriv_ini_file = None, deriv_values_file = None, deriv_values_list_file = None,
                   sbatch_file = None):
    """
    Cleanup == 2 means cleaning everything, cleanup = 1 means only cleaning up the temp deriv folder but keeps the dx values
    """
    kcap_run = kcap_deriv(mock_run = mock_run, 
                          param_to_vary = param_to_vary, 
                          params_to_fix = params_to_fix,
                          vals_to_diff = vals_to_diff,
                          mocks_dir = mocks_dir, 
                          mocks_name = mocks_name,
                          deriv_dir = deriv_dir, 
                          deriv_name = deriv_name, 
                          deriv_ini_file = deriv_ini_file, 
                          deriv_values_file = deriv_values_file, 
                          deriv_values_list_file = deriv_values_list_file,
                          sbatch_file = sbatch_file)
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
            kcap_run.cleanup_dx()

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
        inv_eigenvals = np.zeros(shape = covariance.shape)
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

def get_params(mock_run, vals_to_read, mocks_dir = None, mocks_name = None):
    values_method = read_kcap_values(mock_run = mock_run, mocks_dir = mocks_dir, mocks_name = mocks_name)
    values_read = values_method.get_params(parameter_list = vals_to_read)
    return values_read

def get_fiducial_deriv(deriv_params, data_params, fiducial_run = 0, mocks_dir = None, mocks_name = None, bin_order = None):
    for i, deriv_param in enumerate(deriv_params):
        if i == 0:
            deriv_vals_to_get = [data_param.split("--")[0] + '_' + deriv_param + '_deriv--' + data_param.split("--")[-1] for data_param in data_params]
            deriv_vector_dict = get_values(mock_run = fiducial_run, vals_to_read = deriv_vals_to_get, mocks_dir = mocks_dir, mocks_name = mocks_name, bin_order = bin_order)
            deriv_vector = np.array([])
            for data_deriv_param in deriv_vals_to_get:
                deriv_vector = np.append(deriv_vector, deriv_vector_dict[data_deriv_param])
            deriv_matrix = np.zeros(shape = (len(deriv_params), len(deriv_vector)))
        else:
            deriv_vals_to_get = [data_param.split("--")[0] + '_' + deriv_param + '_deriv--' + data_param.split("--")[-1] for data_param in data_params]
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

    likelihood = np.empty(1)
    for i in range(sim_number):
        try:
            likelihood = np.vstack((likelihood, get_likelihood(mock_run = i, like_name = "loglike_like", mocks_dir = mocks_dir, mocks_name = mocks_name)))
        except:
            print("Mock run %s doesn't exist, skipping this likelihood" % (i))

    return likelihood[1:] 

def get_sim_batch_data_vectors(sim_number, data_params, data_vector_length = 270, mocks_dir = None, mocks_name = None):
    """
    Fetches the data vector
    """
    sim_data_vector = np.empty(data_vector_length)

    for i in range(sim_number):
        try:
            data_vector = get_single_data_vector(mock_run = i, data_params = data_params, mocks_dir = mocks_dir, mocks_name = mocks_name)
            sim_data_vector = np.vstack((sim_data_vector, data_vector))
        except:
            print("Mock run %s doesn't exist, skipping this datavector" % (i))

    print("Fetched values!")

    return sim_data_vector[1:]

def get_sim_batch_thetas(sim_number, theta_names, mocks_dir = None, mocks_name = None):
    """
    Fetches all of the simulation theta values
    """
    thetas = np.empty(len(theta_names))
    for i in range(sim_number):
        try:
            theta = get_params(mock_run = i, vals_to_read = theta_names, mocks_dir = mocks_dir, mocks_name = mocks_name)
            theta = np.array(list(theta.values()))
            thetas = np.vstack((thetas, theta))
        except:
            print("Mock run %s doesn't exist, skipping this theta val" % (i))
    
    return thetas[1:]

def cleanup_folders(mock_run_start, num_mock_runs, mocks_dir = None, mocks_name = None,
                   folders_to_keep = ["cosmological_parameters",
                                      "bias_parameters",
                                      "halo_model_parameters",
                                      "intrinsic_alignment_parameters",
                                      "nofz_shifts",
                                      "theory_data_covariance",
                                      "growth_parameters",
                                      "priors",
                                      "shear_xi_minus_binned", 
                                      "shear_xi_plus_binned", 
                                      "bandpowers",
                                      "shear_pcl",
                                      "shear_pcl_novd"],
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
                   folders_to_keep = ["cosmological_parameters",
                                      "bias_parameters",
                                      "halo_model_parameters",
                                      "intrinsic_alignment_parameters",
                                      "nofz_shifts",
                                      "salmo",
                                      "salmo_novd",
                                      "theory_data_covariance",
                                      "growth_parameters",
                                      "priors",
                                      "shear_xi_minus_binned", 
                                      "shear_xi_plus_binned", 
                                      "bandpowers",
                                      "shear_bias",
                                      "shear_bias_novd",
                                      "shear_pcl",
                                      "shear_pcl_novd"],
                   files_to_remove = ["theory_data_covariance/covariance.txt", "theory_data_covariance/inv_covariance.txt"]):
    print("Initiating unzip and cleanup afterwards... ")
    clean_method = organise_kcap_output(mock_run_start = mock_run_start, num_mock_runs = num_mock_runs, mocks_dir = mocks_dir, mocks_name = mocks_name, 
                                        folders_to_keep = folders_to_keep, files_to_remove = files_to_remove)
    clean_method.extract_and_delete()
    print("Enjoy that sweet sweet disk space and your extracted files!")
    
if __name__ == "__main__":      
    extract_and_cleanup(mock_run_start = 0, num_mock_runs = 100, mocks_dir = '/share/data1/klin/kcap_out/kids_1000_mocks/trial_43_fiducial_cosmology_sensitivity',
                        mocks_name = 'kids_1000_mocks')
   
    
    # data_vectors = get_sim_batch_data_vectors(sim_number = 8000, data_params = ['theory_data_covariance--theory'], data_vector_length = 270, 
    #                                           mocks_dir = '/share/data1/klin/kcap_out/kids_1000_mocks/nz_covariance_testing', mocks_name = 'kids_1000_cosmology_with_nz_shifts_corr')
    
    # np.savetxt('/share/data1/klin/kcap_out/kids_1000_mocks/nz_covariance_testing/data_thetas/data_vectors.txt', data_vectors)
    
    # thetas = get_sim_batch_thetas(sim_number = 8000, theta_names = ['cosmological_parameters--sigma_8', 
    #                                                                 'cosmological_parameters--omch2',
    #                                                                 'intrinsic_alignment_parameters--a',
    #                                                                 'cosmological_parameters--n_s',
    #                                                                 'halo_model_parameters--a',
    #                                                                 'cosmological_parameters--h0',
    #                                                                 'cosmological_parameters--ombh2',
    #                                                                 'nofz_shifts--bias_1',
    #                                                                 'nofz_shifts--bias_2',
    #                                                                 'nofz_shifts--bias_3',
    #                                                                 'nofz_shifts--bias_4',
    #                                                                 'nofz_shifts--bias_5'], 
    #                               mocks_dir = '/share/data1/klin/kcap_out/kids_1000_mocks/nz_covariance_testing', mocks_name = 'kids_1000_cosmology_with_nz_shifts_corr')
    
    # np.savetxt('/share/data1/klin/kcap_out/kids_1000_mocks/nz_covariance_testing/data_thetas/thetas.txt', thetas)

# For regular deriv calcs -----------------------------------------------------------------------------------------------------

    # run_kcap_deriv(mock_run = 0, 
    #             param_to_vary = "cosmological_parameters--ombh2",
    #             params_to_fix = ['cosmological_parameters--sigma_8',
    #                              'intrinsic_alignment_parameters--a',
    #                              'cosmological_parameters--n_s',
    #                              'halo_model_parameters--a',
    #                              'cosmological_parameters--h0',
    #                              'cosmological_parameters--ombh2',
    #                              'nofz_shifts--bias_1',
    #                              'nofz_shifts--bias_2',
    #                              'nofz_shifts--bias_3',
    #                              'nofz_shifts--bias_4',
    #                              'nofz_shifts--bias_5'],
    #             vals_to_diff = ["bandpowers--theory_bandpower_cls", "bandpowers--noisey_bandpower_cls"],
    #             step_size = 0.01,
    #             stencil_pts = 5,
    #             mocks_dir = '/share/data1/klin/kcap_out/kids_1000_glass_mocks/glass_fiducial_and_data',
    #             mocks_name = 'glass_fiducial',
    #             cleanup = 2,
    #             deriv_ini_file = '/share/splinter/klin/kcap_glass/runs/lfi_config/kids_glass_deriv_pipeline.ini', 
    #             deriv_values_file = '/share/splinter/klin/kcap_glass/runs/lfi_config/kids_glass_deriv_values.ini', 
    #             deriv_values_list_file = '/share/splinter/klin/kcap_glass/runs/lfi_config/kids_glass_deriv_values_list.ini',
    #             sbatch_file = '/share/splinter/klin/slurm/slurm_glass_derivs.sh'
    #             )

    # run_omega_m_deriv(mock_run = 0, 
    #                   params_varied = ["cosmological_parameters--omch2"], 
    #                   vals_to_diff = ["bandpowers--theory_bandpower_cls", "bandpowers--noisey_bandpower_cls"], 
    #                   mocks_dir = '/share/data1/klin/kcap_out/kids_1000_glass_mocks/glass_fiducial_and_data', 
    #                   mocks_name = 'glass_fiducial')