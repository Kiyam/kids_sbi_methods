import numpy as np
import configparser as cfg
import subprocess
import os
import glob
import tarfile
import shutil
import time
import pyDOE as pydoe
from scipy.stats.distributions import norm
from pathlib import Path


class read_kcap_values:
    def __init__(self, mock_run=None, mocks_dir=None, mocks_name=None, bin_order=None):
        # mocks_dir settings
        self.mocks_dir = mocks_dir
        self.mocks_name = mocks_name
        self.mock_run = self.check_mock_run_exists(mock_run)

        if self.mock_run is None:
            self.mock_path = f"{self.mocks_dir}/{self.mocks_name}"
        else:
            self.mock_path = f"{self.mocks_dir}/{self.mocks_name}_{self.mock_run}"
        self.content = self.fetch_tar_content()

        if bin_order is None:
            self.bin_order = [
                "bin_1_1",
                "bin_2_1",
                "bin_2_2",
                "bin_3_1",
                "bin_3_2",
                "bin_3_3",
                "bin_4_1",
                "bin_4_2",
                "bin_4_3",
                "bin_4_4",
                "bin_5_1",
                "bin_5_2",
                "bin_5_3",
                "bin_5_4",
                "bin_5_5",
            ]
        else:
            self.bin_order = bin_order

    def check_mock_run_exists(self, mock_run):
        """
        Checks if the requested mock_run file exists, and if not will check for a .tgz file of the same file name and untar as necessary
        """
        if mock_run is None:
            if os.path.exists(f"{self.mocks_dir}/{self.mocks_name}.tgz"):
                return None
            else:
                raise Exception("Sorry, the requested mock run doesn't exist")
        else:
            if os.path.exists(f"{self.mocks_dir}/{self.mocks_name}_{mock_run}.tgz"):
                return str(mock_run)
            else:
                raise Exception(
                    f"Sorry, the requested mock run, {mock_run}, doesn't exist"
                )

    def fetch_tar_content(self):
        """
        Untars file
        """
        try:
            content = tarfile.open(f"{self.mock_path}.tgz", "r:gz")
        except Exception:
            raise Exception("Badly formed tarfile")

        return content

    def read_vals(self, vals_to_read, truncate=0, output_type="as_dict"):
        """
        Options are:
        as_dict - returns the values as a dictionary with the keys matched to the value in the input vals_to_read list
        as_matrix - reshapes the array to a matrix
        as_flat - returns the values as a flat vector
        """
        if isinstance(vals_to_read, list):
            vals_to_read = vals_to_read
        elif isinstance(vals_to_read, str):
            vals_to_read = [vals_to_read]
        else:
            raise Exception(
                "Badly defined values to read, needs to be of type string or list"
            )

        vals_dict = {}
        for val_names in vals_to_read:
            val_folder, val_name = val_names.rsplit("--", 1)
            try:
                if "bin" in val_name:
                    files_list = [
                        bin_file
                        for bin_file in self.content.getnames()
                        if f"{val_folder}/{val_name}" in bin_file
                    ]
                    bin_names = [
                        bin_name.split("/")[-1].replace(".txt", "")
                        for bin_name in files_list
                    ]

                    bin_vals_dict = {}
                    for i, file_name in enumerate(files_list):
                        bin_vals_dict[bin_names[i]] = np.loadtxt(
                            self.content.extractfile(file_name)
                        )[truncate:]

                    vals_array = np.array([])
                    for bin_name in self.bin_order:
                        try:
                            vals_array = np.append(vals_array, bin_vals_dict[bin_name])
                        except Exception:
                            vals_array = np.append(
                                vals_array,
                                bin_vals_dict[
                                    [x for x in bin_vals_dict.keys() if bin_name in x][
                                        0
                                    ]
                                ],
                            )
                else:
                    try:
                        file_name = [
                            file_name
                            for file_name in self.content.getnames()
                            if f"{val_folder}/{val_name}" in file_name
                        ][0]
                        vals_array = np.loadtxt(self.content.extractfile(file_name))[
                            truncate:
                        ]
                    except Exception:
                        raise Exception("Badly defined parameter name %s" % val_names)
            except Exception:
                raise Exception("Badly defined parameter name %s" % val_names)

            vals_dict[val_names] = vals_array

        if output_type == "as_dict":
            return vals_dict
        elif output_type == "as_flat":
            data_vector = np.array(list(vals_dict.values())).flatten()
            return data_vector
        elif output_type == "as_matrix":
            data_vector = np.array(list(vals_dict.values()))
            return data_vector

    def read_params(self, parameter_list, output_type="as_dict"):
        """
        Gets parameters from the specified mock run
        """
        parameter_dict = {}  # Dictionary of parameters
        for param_name in parameter_list:
            header, name = param_name.split("--")
            try:
                file_loc = [
                    file_name
                    for file_name in self.content.getnames()
                    if f"{header}/values.txt" in file_name
                ][0]
            except Exception:
                raise Exception(f"Badly defined parameter name {param_name}")
            param_val = None
            for line in self.content.extractfile(file_loc):
                key, value = [word.strip() for word in line.decode("utf-8").split("=")]
                if key == name:
                    param_val = float(value)
            if param_val is not None:
                parameter_dict[param_name] = param_val
            else:
                raise Exception(f"{param_name} missing value")
        print(f"Fetched parameters are: {parameter_dict}")

        if output_type == "as_dict":
            return parameter_dict
        elif output_type == "as_flat":
            param_vector = np.array(list(parameter_dict.values())).flatten()
            return param_vector

    def write_to_tar(self, to_write, to_write_headers, file_loc):
        """
        Adds a list of files to the tar
        """
        self.content.extractall("/")
        self.content.close()
        print("Writing values to tar...")
        assert isinstance(to_write, list), "to_write needs to be a list"
        assert isinstance(to_write_headers, list), "to_write_headers needs to be a list"
        assert isinstance(file_loc, list), "file_loc needs to be a list"
        assert (
            len(to_write) == len(file_loc) and len(to_write) == len(file_loc)
        ), "Amount of data to write doesn't match either the number of file locations or headers provided"
        for i in range(len(to_write)):
            file_path = f"{self.mock_path}/{file_loc[i]}"
            if file_path[-1] == "/":  # make sure the parent directory exists
                Path(file_path).mkdir(parents=True, exist_ok=True)
            else:
                Path("/" + "/".join(file_path.split("/")[:-1])).mkdir(
                    parents=True, exist_ok=True
                )
            np.savetxt(file_path, to_write[i], header=to_write_headers[i])

        new_files_list = glob.glob(self.mock_path + "/*/*")
        content = tarfile.open(self.mock_path + ".tgz", "w:gz")
        for file in new_files_list:
            content.add(file)
        content.close()

        shutil.rmtree(self.mock_path)

        self.content = tarfile.open(self.mock_path + ".tgz", "r:gz")

    def del_from_tar(self, files_to_delete):
        """
        Deletes a specific file from within the tar
        """
        self.content.extractall()
        self.content.close()

        assert isinstance(files_to_delete, list), "files_to_delete needs to be a list"

        for file in files_to_delete:
            os.remove(self.mock_path + "/" + file)

        new_files_list = glob.glob(self.mock_path + "/*/*")
        content = tarfile.open(self.mock_path + ".tgz", "w:gz")
        for file in new_files_list:
            content.add(file)
        content.close()

        self.content = tarfile.open(self.mock_path + ".tgz", "r:gz")

    def close(self):
        """
        Cleanly closes tarfile after reading
        """
        self.content.close()

    def delete_mock_tgz(self, mock_run):
        tgz_to_delete = (
            self.mocks_dir + "/" + self.mocks_name + "_" + str(mock_run) + ".tgz"
        )
        os.remove(tgz_to_delete)
        print("Removed %s" % str(tgz_to_delete))

    def delete_all_tgz(self):
        all_tgz = glob.glob(self.mocks_dir + "/" + self.mocks_name + "*.tgz")
        for tgz_file in all_tgz:
            os.remove(tgz_file)
            print("Removed %s" % str(tgz_file))
        print("All .tgz files removed!")


class kcap_deriv(read_kcap_values):
    def __init__(
        self,
        param_to_vary,
        params_to_fix,
        vals_to_diff,
        stencil_pts=5,
        step_size=0.01,
        bin_order=None,
        mock_run=None,
        mocks_dir=None,
        mocks_name=None,
        deriv_dir=None,
        deriv_name=None,
        deriv_ini_file=None,
        deriv_values_file=None,
        deriv_values_list_file=None,
        sbatch_file=None,
    ):
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
            self.mock_path = "{0}/{1}".format(self.mocks_dir, self.mocks_name)
        else:
            self.mock_path = "{0}/{1}_{2}".format(
                self.mocks_dir, self.mocks_name, self.mock_run
            )

        print(self.mock_path)

        self.content = self.fetch_tar_content()
        self.stencil_pts = stencil_pts
        self.step_size = step_size

        if param_to_vary in params_to_fix:
            raise Exception(
                "Specified parameter to vary is also specified to not vary, inconsistent settings"
            )

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

        if bin_order is None:
            self.bin_order = [
                "bin_1_1",
                "bin_2_1",
                "bin_2_2",
                "bin_3_1",
                "bin_3_2",
                "bin_3_3",
                "bin_4_1",
                "bin_4_2",
                "bin_4_3",
                "bin_4_4",
                "bin_5_1",
                "bin_5_2",
                "bin_5_3",
                "bin_5_4",
                "bin_5_5",
            ]
        else:
            self.bin_order = bin_order

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

        dx_array = np.arange(self.stencil_pts)
        middle_index = int((self.stencil_pts - 1) / 2)
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
            if middle_val == 0.0:
                self.abs_step_size = self.step_size
            else:
                self.abs_step_size = np.absolute(middle_val) * self.step_size
            vals_array = middle_val + dx_array * self.abs_step_size

            new_param_string = (
                str(vals_array[0]) + " " + str(middle_val) + " " + str(vals_array[-1])
            )
            values_config[self.param_header][name] = new_param_string
            file_text = ["#" + self.param_header + "--" + name + "\n"]
            for val in vals_array:
                file_text.append(str(val) + "\n")

            values_list_file = open(self.deriv_values_list, "w")
            values_list_file.writelines(file_text)
            values_list_file.close()

        else:
            raise Exception("Badly defined parameter to vary...")

        with open(self.deriv_values_file, "w") as configfile:
            values_config.write(configfile)

    def run_deriv_kcap(self, mpi_opt=False, cluster=True):
        if mpi_opt is True:
            subprocess.run(
                [
                    "mpirun",
                    "-n",
                    str(self.stencil_pts - 1),
                    "--use-hwthread-cpus",
                    "cosmosis",
                    "--mpi",
                    self.deriv_ini_file,
                ]
            )
        elif cluster is True:
            output = subprocess.run(
                ["sbatch", self.sbatch_file], stdout=subprocess.PIPE, text=True
            )
            text_out = output.stdout
            jobid = text_out.split(" ")[-1].rstrip()
            print("------------------------")
            print("Submitted slurm job with jobid: " + jobid)
            print("------------------------")
            return jobid
        elif mpi_opt is False:
            subprocess.run(["cosmosis", self.deriv_ini_file])
        else:
            raise Exception("Failed to initialise cosmosis pipeline for derivatives")

    def poll_cluster_finished(self, jobid):
        start_time = time.time()
        elapsed = time.time() - start_time
        finished = False
        while elapsed <= 172800.0 and finished is not True:
            try:
                subprocess.check_output(["squeue", "-j", jobid])
                time.sleep(10)
            except:
                print("Simulations finished!")
                finished = True
        print("Waiting to ensure all IO operations are finished")
        time.sleep(10)

    def fetch_dx_values(self):
        dx_list = []
        num_deriv_pts = self.stencil_pts - 1
        if len(glob.glob(f"{self.deriv_dir}/{self.deriv_name}_*.tgz")) == num_deriv_pts:
            for deriv_run in range(self.stencil_pts - 1):
                deriv_tar = read_kcap_values(
                    mock_run=deriv_run,
                    mocks_dir=self.deriv_dir,
                    mocks_name=self.deriv_name,
                )
                vals_dict = deriv_tar.read_vals(vals_to_read=self.vals_to_diff)
                dx_list.append(vals_dict)
        else:
            raise Exception(
                "Sorry, you seem to be missing the derivatives, try running KCAP for the requested derivative steps first"
            )

        dx_dict = {}
        for val_to_diff in self.vals_to_diff:
            val_array = np.array([])
            for i in range(num_deriv_pts):
                val_array = np.append(val_array, dx_list[i][val_to_diff])
            val_array = val_array.reshape(num_deriv_pts, -1)
            dx_dict[val_to_diff] = val_array

        return dx_dict

    def calc_deriv_values(self):
        """
        Calculates the first derivative using a 5 point stencil
        """

        if self.stencil_pts == 3:
            stencil_coeffs = np.array([-1 / 2, 1 / 2])
        elif self.stencil_pts == 5:
            stencil_coeffs = np.array([1 / 12, -2 / 3, 2 / 3, -1 / 12])
        elif self.stencil_pts == 7:
            stencil_coeffs = np.array([-1 / 60, 3 / 20, -3 / 4, 3 / 4, -3 / 20, 1 / 60])
        elif self.stencil_pts == 9:
            stencil_coeffs = np.array(
                [1 / 280, -4 / 105, 1 / 5, -4 / 5, 4 / 5, -1 / 5, 4 / 105, -1 / 280]
            )
        else:
            raise Exception("Invalid stencil number inputted")

        dx_dict = self.fetch_dx_values()
        print(
            "All values needed for %s derivatives wrt to %s found, now calculating..."
            % (self.vals_to_diff, self.param_to_vary)
        )
        deriv_dict = {}
        for val_to_diff in self.vals_to_diff:
            first_deriv_vals = (
                np.dot(stencil_coeffs, dx_dict[val_to_diff]) / self.abs_step_size
            )
            deriv_dict[val_to_diff] = first_deriv_vals
        print("Derivatives calculated succesfully")

        return deriv_dict

    def save_deriv_values(self, deriv_dict):
        to_write = []
        to_write_headers = []
        file_locs = []
        print(self.mock_path)
        for val_to_diff in self.vals_to_diff:
            val_to_diff_head, val_to_diff_name = val_to_diff.split(sep="--")
            if (
                "bin" in val_to_diff
            ):  # This option saves the files that should be binned into their respective bin folders to maintain folder/file structure
                deriv_vals = deriv_dict[val_to_diff].reshape((len(self.bin_order), -1))
                for i, vals in enumerate(deriv_vals):
                    to_write.append(vals)
                    to_write_headers.append(
                        f"deriv values for {val_to_diff} wrt {self.param_to_vary} with relative stepsize of {self.step_size} and absolute stepsize of {self.abs_step_size}"
                    )
                    file_locs.append(
                        f"{val_to_diff_head}_{self.param_to_vary}_deriv/{self.bin_order[i]}.txt"
                    )
            else:
                to_write.append(deriv_dict[val_to_diff])
                to_write_headers.append(
                    f"deriv values for {val_to_diff} wrt {self.param_to_vary} with relative stepsize of {self.step_size} and absolute stepsize of {self.abs_step_size}"
                )
                file_locs.append(
                    f"{val_to_diff_head}_{self.param_to_vary}_deriv/{val_to_diff_name}.txt"
                )
        self.write_to_tar(to_write, to_write_headers, file_locs)

    def check_existing_derivs(self):
        print("Checking if the corresponding derivatives exist...")
        for val_to_diff in self.vals_to_diff:
            val_to_diff_head, val_to_diff_name = val_to_diff.split(sep="--")
            if "bin" in val_to_diff_name:
                num_base = len(self.bin_order)
                num_found = len(
                    [
                        bin_file
                        for bin_file in self.content.getnames()
                        if f"{val_to_diff_head}_{self.param_to_vary}_deriv/{val_to_diff_name}"
                        in bin_file
                    ]
                )
            else:
                num_base = len(
                    [
                        files
                        for files in self.content.getnames()
                        if f"{val_to_diff_head}/{val_to_diff_name}" in files
                    ]
                )
                num_found = len(
                    [
                        files
                        for files in self.content.getnames()
                        if f"{val_to_diff_head}_{self.param_to_vary}_deriv/{val_to_diff_name}"
                        in files
                    ]
                )
            if num_base == num_found:
                print(
                    f"Files for {val_to_diff_head} numerical derivative values wrt to {self.param_to_vary} found."
                )
                pass
            else:
                print(
                    f"Missing derivatives for {val_to_diff_head} wrt to {self.param_to_vary}."
                )
                return False

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
                raise Exception(
                    "Error during directory cleanup, please manually inspect!"
                )
        else:
            raise Exception(
                "Not all files found, exiting cleanup. Please manually inspect!"
            )

    def omega_m_deriv(self):
        h = self.read_params(
            parameter_list=["cosmological_parameters--h0"], output_type="as_flat"
        )

        omch2_deriv_names = []
        for (
            val_to_diff
        ) in self.vals_to_diff:  # make a list of the names of the variables to fetch
            val_to_diff_head, val_to_diff_name = val_to_diff.split(sep="--")
            omch2_deriv_names.append(
                val_to_diff_head
                + "_"
                + self.param_to_vary
                + "_deriv--"
                + val_to_diff_name
            )

        omch_deriv = self.read_vals(vals_to_read=omch2_deriv_names)
        to_write = []
        to_write_headers = []
        file_locs = []

        for i, val_to_diff in enumerate(self.vals_to_diff):
            val_to_diff_head, val_to_diff_name = val_to_diff.split(sep="--")
            omega_m_deriv_vals = omch_deriv[omch2_deriv_names[i]] * (h**2)

            if (
                "bin" in val_to_diff
            ):  # This option saves the files that should be binned into their respective bin folders to maintain folder/file structure
                deriv_vals = omega_m_deriv_vals.reshape((len(self.bin_order), -1))
                for i, vals in enumerate(deriv_vals):
                    to_write.append(vals)
                    to_write_headers.append(
                        "deriv values for {0} wrt {1} derived from omch2 deriv".format(
                            val_to_diff, self.param_to_vary
                        )
                    )
                    file_locs.append(
                        val_to_diff_head
                        + "_cosmological_parameters--omega_m_deriv/"
                        + self.bin_order[i]
                        + ".txt"
                    )
            else:
                to_write.append(omega_m_deriv_vals)
                to_write_headers.append(
                    "deriv values for {0} wrt {1} derived from omch2 deriv".format(
                        val_to_diff, self.param_to_vary
                    )
                )
                file_locs.append(
                    val_to_diff_head
                    + "_cosmological_parameters--omega_m_deriv/"
                    + val_to_diff_name
                    + ".txt"
                )

            print(
                "All values needed for %s derivatives wrt to omega_m calculated, saving to file..."
                % (val_to_diff)
            )
            self.write_to_tar(to_write, to_write_headers, file_locs)
            print("Derivatives saved succesfully")


class kcap_delfi:
    def __init__(
        self,
        params_to_vary,
        params_to_read,
        data_name,
        data_vec_length,
        mocks_dir=None,
        mocks_name=None,
        mocks_ini_file=None,
        mocks_values_file=None,
        save_folder=None,
        nz_indices=None,
        nz_cov=None,
        verbose=False,
        slurm_file=None,
        bin_order=None,
    ):
        """
        Gets variable from .env file
        """

        if slurm_file:
            self.is_cluster = True
            self.slurm_file = slurm_file
        else:
            self.is_cluster = False

        # mocks settings
        self.mocks_dir = mocks_dir
        self.mocks_name = mocks_name
        self.kids_pipeline_ini_file = mocks_ini_file
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
        self.verbose = verbose

        if bin_order == None:
            self.bin_order = [
                "bin_1_1",
                "bin_2_1",
                "bin_2_2",
                "bin_3_1",
                "bin_3_2",
                "bin_3_3",
                "bin_4_1",
                "bin_4_2",
                "bin_4_3",
                "bin_4_4",
                "bin_5_1",
                "bin_5_2",
                "bin_5_3",
                "bin_5_4",
                "bin_5_5",
            ]
        else:
            self.bin_order = bin_order

        print("Simulator succesfully initialized")

    def run_delfi_kcap(self, rank=None):
        if self.is_cluster == True:
            output = subprocess.run(
                ["sbatch", self.slurm_file], stdout=subprocess.PIPE, text=True
            )
        else:
            pipeline_file = self.kids_pipeline_ini_file + "_" + str(rank) + ".ini"
            output = subprocess.run(["cosmosis", pipeline_file])

        text_out = output.stdout
        jobid = text_out.split(" ")[-1].rstrip()
        print("------------------------")
        print("Submitted slurm job with jobid: " + jobid)
        print("------------------------")
        return jobid

    def write_delfi_kcap_values(self, theta, rank=None):
        """
        Modifies the values_list.ini file
        """
        if theta.ndim > 1:
            assert theta.shape[1] == len(
                self.params_to_vary
            ), "Different number of parameters to vary fed vs. defined to vary."
        else:
            assert len(theta) == len(
                self.params_to_vary
            ), "Different number of parameters to vary fed vs. defined to vary."
        values_config = cfg.ConfigParser()
        if rank:
            values_list_file = self.kids_pipeline_values_file + "_" + str(rank) + ".ini"
        else:
            values_list_file = self.kids_pipeline_values_file + ".ini"

        # The NZ mixing stuff
        if self.nz_indices:
            inv_L = np.linalg.inv(np.linalg.cholesky(self.nz_cov))
            if theta.ndim > 1:
                nz_theta = theta[:, min(self.nz_indices) : max(self.nz_indices) + 1]
                nz_theta = np.dot(inv_L, nz_theta.T).T
                theta[:, min(self.nz_indices) : max(self.nz_indices) + 1] = nz_theta
            else:
                nz_theta = theta[min(self.nz_indices) : max(self.nz_indices) + 1]
                nz_theta = np.dot(inv_L, nz_theta)
                theta[min(self.nz_indices) : max(self.nz_indices) + 1] = nz_theta

        # Just doing some parameter renaming
        params_to_vary = self.params_to_vary
        for i, val in enumerate(params_to_vary):
            if val == "cosmological_parameters--sigma_8":
                params_to_vary[i] = "cosmological_parameters--sigma_8_input"
            elif val == "cosmological_parameters--s_8":
                params_to_vary[i] = "cosmological_parameters--s_8_input"

        # Write the values list to file
        header = " ".join(params_to_vary)
        if theta.ndim > 1:
            np.savetxt(fname=values_list_file, X=theta, header=header)
        else:
            np.savetxt(fname=values_list_file, X=np.array([theta]), header=header)

        print("Succesfully set values to run simulations on")

    def poll_cluster_finished(self, jobid):
        start_time = time.time()
        last_sim_time = time.time()
        finished = False
        num_done = 0
        while time.time() - start_time <= 172800.0 and finished != True:
            try:
                subprocess.check_output(["squeue", "-j", jobid])
                if (
                    len(glob.glob(self.mocks_dir + "/" + self.mocks_name + "_*.tgz"))
                    > num_done
                ):
                    num_done = len(
                        glob.glob(self.mocks_dir + "/" + self.mocks_name + "_*.tgz")
                    )
                    last_sim_time = time.time()
                if time.time() - last_sim_time > 2400 and num_done != 0:
                    subprocess.run(["scancel", jobid])
                time.sleep(10)
            except:
                print("Simulations finished!")
                finished = True
        print("Waiting to ensure all IO operations are finished")
        time.sleep(10)

    def save_sims(self, sim_data_vector, thetas):
        if self.save_folder:
            which_population = len(glob.glob(self.save_folder + "/*/"))
            os.makedirs(self.save_folder + "/population_" + str(which_population))
            np.savetxt(
                self.save_folder + "/data_population_" + str(which_population) + ".txt",
                sim_data_vector,
            )
            np.savetxt(
                self.save_folder
                + "/thetas_population_"
                + str(which_population)
                + ".txt",
                thetas,
            )
            for mocks_file in glob.glob(self.mocks_dir + "/*.tgz"):
                dest = (
                    self.save_folder
                    + "/population_"
                    + str(which_population)
                    + "/"
                    + mocks_file.split("/")[-1][:-4]
                    + ".tgz"
                )
                shutil.move(mocks_file, dest, copy_function=shutil.copytree)
            shutil.rmtree(self.mocks_dir)

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

        thetas, sim_data_vector = None, None

        for i in range(self.num_mock_runs):
            self.mock_run = str(i)
            try:
                content = read_kcap_values(
                    mock_run=i,
                    mocks_dir=self.mocks_dir,
                    mocks_name=self.mocks_name,
                    bin_order=self.bin_order,
                )
                data_vector = content.read_vals(
                    vals_to_read=self.data_name, output_type="as_flat"
                )
                theta = content.read_params(
                    parameter_list=self.params_to_read, output_type="as_flat"
                )
                if thetas is not None and sim_data_vector is not None:
                    thetas = np.vstack((thetas, theta))
                    sim_data_vector = np.vstack((sim_data_vector, data_vector))
                else:
                    thetas = theta
                    sim_data_vector = data_vector
            except:
                print("Mock run %s doesn't exist, skipping this datavector" % (i))

        assert len(sim_data_vector) == len(thetas), (
            "Mismatch between number of fetched simulation data vectors: %s and parameter sets: %s"
            % (len(sim_data_vector), len(thetas))
        )

        self.save_sims(sim_data_vector, thetas)
        try:
            self.clean_mocks_folder()
        except:
            print("Mock folder not found, already cleaned up")

        return sim_data_vector, thetas


class kcap_delfi_proposal:
    def __init__(
        self,
        n_initial,
        lower,
        upper,
        transformation=None,
        delta_z_indices=None,
        delta_z_cov=None,
        factor_of_safety=5,
        iterations=1000,
    ):
        # The lower and upper bounds for this are for a truncated Gaussian, so it should be different to the PyDELFI prior
        assert len(lower) == len(upper)
        num_samples = n_initial * factor_of_safety
        points = pydoe.lhs(
            len(lower), samples=num_samples, criterion="cm", iterations=iterations
        )
        for i in range(len(lower)):
            if (
                delta_z_indices is not None and i in delta_z_indices
            ):  # Does a truncated Gaussian
                p = 0
                while p == 0:
                    x = norm(loc=0, scale=1).ppf(points[:, i])
                    p = self.gaussian(lower[i], upper[i], x)
                points[:, i] = x
            else:
                val_range = upper[i] - lower[i]
                points[:, i] *= val_range
                points[:, i] += lower[i]

        if transformation is not None:
            points = transformation(points)

        if delta_z_cov is not None:
            uncorr_nz = points[:, min(delta_z_indices) : max(delta_z_indices) + 1]
            L = np.linalg.cholesky(delta_z_cov)
            corr_nz = np.inner(L, uncorr_nz).T
            points[:, min(delta_z_indices) : max(delta_z_indices) + 1] = corr_nz

        self.sample_points = iter(points)

    def gaussian(self, g_lower, g_upper, x):
        inrange = np.prod(x > g_lower) * np.prod(x < g_upper)
        return inrange * np.prod(g_upper - g_lower)

    def draw(self):
        return next(self.sample_points)


def run_kcap_deriv(
    param_to_vary,
    params_to_fix,
    vals_to_diff,
    step_size,
    stencil_pts,
    mock_run=None,
    mocks_dir=None,
    mocks_name=None,
    cleanup=True,
    deriv_dir=None,
    deriv_name=None,
    deriv_ini_file=None,
    deriv_values_file=None,
    deriv_values_list_file=None,
    sbatch_file=None,
):
    """
    Cleanup == 2 means cleaning everything, cleanup = 1 means only cleaning up the temp deriv folder but keeps the dx values
    """
    kcap_run = kcap_deriv(
        param_to_vary=param_to_vary,
        params_to_fix=params_to_fix,
        vals_to_diff=vals_to_diff,
        stencil_pts=stencil_pts,
        step_size=step_size,
        mock_run=mock_run,
        mocks_dir=mocks_dir,
        mocks_name=mocks_name,
        deriv_dir=deriv_dir,
        deriv_name=deriv_name,
        deriv_ini_file=deriv_ini_file,
        deriv_values_file=deriv_values_file,
        deriv_values_list_file=deriv_values_list_file,
        sbatch_file=sbatch_file,
    )

    check = kcap_run.check_existing_derivs()
    if check is True:
        print(
            "All files found for these parameters, skipping this particular deriv run"
        )
    else:
        print("Not all values found, continuing script...")
        kcap_run.write_deriv_ini_values()
        jobid = kcap_run.run_deriv_kcap(mpi_opt=False, cluster=True)
        kcap_run.poll_cluster_finished(jobid=jobid)
        deriv_dict = kcap_run.calc_deriv_values()
        kcap_run.save_deriv_values(deriv_dict)
        if param_to_vary == "cosmological_parameters--omch2":
            kcap_run.omega_m_deriv()
        if cleanup == True:
            kcap_run.cleanup_deriv_folder()


def run_omega_m_deriv(
    mock_run, params_varied, vals_to_diff, mocks_dir=None, mocks_name=None
):
    kcap_run = kcap_deriv(
        mock_run=mock_run,
        param_to_vary="cosmological_parameters--omega_m",
        params_to_fix=[None],
        vals_to_diff=vals_to_diff,
        mocks_dir=mocks_dir,
        mocks_name=mocks_name,
    )
    check = kcap_run.check_existing_derivs()
    if check is True:
        print(
            "All files found for these parameters, skipping this particular deriv run"
        )
    else:
        print("Not all values found, continuing script...")
        kcap_run.first_omega_m_deriv(params_varied)


def calc_inv_covariance(covariance, method="eigen"):
    if method == "symmetrised":
        # Averaging the inverse covariance
        inv_covariance = np.linalg.inv(covariance)
        inv_covariance = (inv_covariance + inv_covariance.T) / 2
    elif method == "cholesky":
        # Cholesky method for calculating the inverse covariance
        cholesky_decom = np.linalg.cholesky(covariance)
        identity = np.identity(len(covariance))
        y = np.linalg.solve(cholesky_decom, identity)
        inv_covariance = np.linalg.solve(cholesky_decom.T, y)
    elif method == "eigen":
        # Eigenvalue decomposition method of calculating the inverse covariance
        eigenvals, eigenvectors = np.linalg.eig(covariance)
        inv_eigenvals = np.zeros(shape=covariance.shape)
        for i, val in enumerate(eigenvals):
            if val > 0.0:
                inv_eigenvals[i][i] += 1 / val
            else:
                pass
        inv_covariance = np.dot(eigenvectors, np.dot(inv_eigenvals, eigenvectors.T))
    elif method == "suppressed":
        for i, cov_slice in enumerate(covariance):
            for j, val in enumerate(cov_slice):
                covariance[i][j] = val * 0.9 ** (abs(i - j))
        inv_covariance = np.inv(covariance)
    else:
        inv_covariance = np.linalg.inv(covariance)

    return inv_covariance


def get_fiducial_deriv(
    deriv_params,
    data_params,
    fiducial_run=None,
    mocks_dir=None,
    mocks_name=None,
    bin_order=None,
    truncate=0,
):
    content = read_kcap_values(
        mock_run=fiducial_run,
        mocks_dir=mocks_dir,
        mocks_name=mocks_name,
        bin_order=bin_order,
    )
    deriv_vals_to_get = []
    for i, data_param in enumerate(data_params):
        for j, deriv_param in enumerate(deriv_params):
            deriv_vals_to_get.append(
                data_param.split("--")[0]
                + "_"
                + deriv_param
                + "_deriv--"
                + data_param.split("--")[-1]
            )

    deriv_matrix = content.read_vals(
        vals_to_read=deriv_vals_to_get, output_type="as_matrix", truncate=truncate
    )
    content.close()
    return deriv_matrix


def get_sim_batch_data_vectors(
    sim_number, data_names, mocks_dir=None, mocks_name=None, bin_order=None, truncate=0
):
    """
    Fetches the data vector
    """
    for i in range(sim_number):
        try:
            content = read_kcap_values(
                mock_run=i,
                mocks_dir=mocks_dir,
                mocks_name=mocks_name,
                bin_order=bin_order,
            )
            data_vector = content.read_vals(
                vals_to_read=data_names, output_type="as_flat", truncate=truncate
            )
            content.close()
            try:
                sim_data_vector = np.vstack((sim_data_vector, data_vector))
            except:
                sim_data_vector = data_vector
            print("Fetched from mock run %s" % (i))
        except:
            print("Mock run %s doesn't exist, skipping this datavector" % (i))

    print("Fetched values!")
    return sim_data_vector


def get_sim_batch_params(
    sim_number, param_names, mocks_dir=None, mocks_name=None, bin_order=None
):
    """
    Fetches all of the simulation theta values
    """
    for i in range(sim_number):
        try:
            content = read_kcap_values(
                mock_run=i,
                mocks_dir=mocks_dir,
                mocks_name=mocks_name,
                bin_order=bin_order,
            )
            param_vector = content.read_params(
                parameter_list=param_names, output_type="as_flat"
            )
            content.close()
            try:
                sim_param_vector = np.vstack((sim_param_vector, param_vector))
            except:
                sim_param_vector = param_vector
            print("Fetched from mock run %s" % (i))
        except:
            print("Mock run %s doesn't exist, skipping this datavector" % (i))

    print("Fetched values!")
    return sim_param_vector


def get_sim_batch_data_params(
    sim_number,
    param_names,
    data_names,
    mocks_dir=None,
    mocks_name=None,
    bin_order=None,
    truncate=0,
):
    """
    Fetches batch of data vectors and parameter values in one go
    """
    for i in range(sim_number):
        try:
            content = read_kcap_values(
                mock_run=i,
                mocks_dir=mocks_dir,
                mocks_name=mocks_name,
                bin_order=bin_order,
            )
            data_vector = content.read_vals(
                vals_to_read=data_names, output_type="as_dict", truncate=truncate
            )
            param_vector = content.read_params(
                parameter_list=param_names, output_type="as_flat"
            )
            content.close()
            try:
                sim_param_vector = np.vstack((sim_param_vector, param_vector))
                for data_name in data_names:
                    sim_data_vector[data_name] = np.vstack(
                        (sim_data_vector[data_name], data_vector[data_name])
                    )
            except:
                sim_data_vector = data_vector
                sim_param_vector = param_vector
            print("Fetched from mock run %s" % (i))
        except:
            print("Mock run %s doesn't exist, skipping this datavector" % (i))

    print("Fetched values!")
    return sim_data_vector, sim_param_vector


def make_tar(mocks_path, cleanup=False):
    new_files_list = glob.glob(mocks_path + "/*/*")
    content = tarfile.open(mocks_path + ".tgz", "w:gz")
    for file in new_files_list:
        content.add(file)
    content.close()
    if cleanup == True:
        if os.path.isfile(mocks_path) or os.path.islink(mocks_path):
            os.remove(mocks_path)  # remove the file
        elif os.path.isdir(mocks_path):
            shutil.rmtree(mocks_path)  # remove dir and all contains


def move_tar(source, dest, copy=False):
    content = tarfile.open(source + ".tgz", "r:gz")
    content.extractall("/")
    # source = source+'.tgz'
    # dest = dest+'.tgz'
    shutil.move(source, dest)
    make_tar(dest, cleanup=True)
    if copy == False:
        if os.path.isfile(source + ".tgz") or os.path.islink(source + ".tgz"):
            os.remove(source + ".tgz")  # remove the file
        elif os.path.isdir(source):
            shutil.rmtree(source)  # remove dir and all contains
