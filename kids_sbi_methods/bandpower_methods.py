from . import kcap_methods as km
import numpy as np
from scipy.stats import wishart

bin_order = []
for i in range(5):
    for j in range(i + 1):
        bin_order.append("bin_%d_%d" % (i + 1, j + 1))

# ell = np.genfromtxt(files_root + 'ell.txt')[1:]
ellip_disp = np.array([0.27, 0.26, 0.27, 0.25, 0.27])
n_eff = (
    np.array([0.62, 1.18, 1.85, 1.26, 1.31]) * (60**2) * (180**2) / (np.pi**2)
)  # in steradian converted from arcmin


def get_cls(l_mode_index=0, num_tom=5, mocks_dir=None, mocks_name=None):
    fid_tar = km.read_kcap_values(mocks_dir=mocks_dir, mocks_name=mocks_name)
    cls_raw = fid_tar.read_vals(
        vals_to_read="shear_cl--bin", output_type="as_flat", truncate=0
    ).reshape(-1, len(ell))
    fid_tar.close()

    bin_order = []
    for i in range(5):
        for j in range(i + 1):
            bin_order.append("bin_%d_%d" % (i + 1, j + 1))

    cls = np.zeros((num_tom, num_tom))

    for i in range(num_tom):
        for j in range(num_tom):
            if j > i:
                cl = cls_raw[bin_order.index("bin_%s_%s" % (j + 1, i + 1))][
                    l_mode_index
                ]
            else:
                cl = cls_raw[bin_order.index("bin_%s_%s" % (i + 1, j + 1))][
                    l_mode_index
                ]
            cls[i, j] = cl

    return cls


def resample_cls(ell, cls, l_mode_index):
    norm = 2 * ell[l_mode_index] + 1
    cls_resampled = wishart.rvs(df=norm, scale=cls / norm)

    return cls_resampled


def bin_cls(cls, num_tom=5):
    cl_binned = {}

    for i in range(num_tom):
        for j in range(num_tom):
            if j <= i:
                cl_binned["bin_%s_%s" % (i + 1, j + 1)] = cls[i, j]
            else:
                pass

    return cl_binned


def add_noise(
    cls,
    bin_1,
    bin_2,
    ellip_disp=np.array([0.27, 0.26, 0.27, 0.25, 0.27]),
    gal_num=np.array([0.62, 1.18, 1.85, 1.26, 1.31]) * (60**2) * (180**2) / (np.pi**2),
):
    try:
        cl = cls["bin_{}_{}".format(bin_1, bin_2)]
    except:
        cl = cls["bin_{}_{}".format(bin_2, bin_1)]

    if bin_1 == bin_2:
        cl += (ellip_disp[int(bin_1) - 1] ** 2) / (2 * gal_num[int(bin_1) - 1])

    return cl


def cls_calc_covariance(cls, ell, l_mode_index, A=1006):
    pre_factor = 2 * np.pi / (A * ell[l_mode_index])
    bin_keys = list(cls.keys())

    cl_ref = [["null" for _ in range(len(bin_keys))] for _ in range(len(bin_keys))]
    for i, bin_1 in enumerate(bin_keys):
        for j, bin_2 in enumerate(bin_keys):
            cl_ref[i][j] = (
                bin_1.split("_")[-2]
                + bin_1.split("_")[-1]
                + bin_2.split("_")[-2]
                + bin_2.split("_")[-1]
            )

    cl_cov = np.zeros((len(cls), len(cls)))

    for row in range(len(cls)):
        for col in range(len(cls)):
            i = cl_ref[row][col][0]
            j = cl_ref[row][col][1]
            k = cl_ref[row][col][2]
            l = cl_ref[row][col][3]

            cl_cov[row][col] = add_noise(cls, i, k) * add_noise(cls, j, l) + add_noise(
                cls, i, l
            ) * add_noise(cls, j, k)

    cl_cov *= pre_factor

    return cl_cov


def bandpower_cov(ell, num_tom=5, num_bands=8, mocks_dir=None, mocks_name=None):
    total_num_ell = len(ell)
    # cl_per_band = int(total_num_ell/num_bands)
    num_pairs = np.sum(np.arange(num_tom + 1))
    temp_cl_cov = np.zeros((total_num_ell, num_pairs, num_pairs))

    for i in range(total_num_ell):
        theory_cls = get_cls(
            l_mode_index=i, num_tom=5, mocks_dir=mocks_dir, mocks_name=mocks_name
        )
        cl_binned = bin_cls(cls=theory_cls, num_tom=5)
        temp_cl_cov[i] += cls_calc_covariance(cls=cl_binned, ell=ell, l_mode_index=i)

    print(temp_cl_cov.shape)
    band_cutoffs = np.logspace(
        np.log(min(ell)), np.log(max(ell) + 1), num_bands + 1, base=np.e
    )
    combined_bandpower_cov = np.zeros((120, 120))
    for i in range(num_bands):
        bandpower_cov = np.zeros((num_pairs, num_pairs))
        for j in range(num_bands):
            bandpower_cov += temp_cl_cov[
                np.logical_and(ell >= band_cutoffs[j], ell < band_cutoffs[j + 1])
            ].sum(axis=0)
        combined_bandpower_cov[
            i * num_pairs : i * num_pairs + num_pairs,
            i * num_pairs : i * num_pairs + num_pairs,
        ] += bandpower_cov

    return combined_bandpower_cov


def bandpower_integral(x, y):
    """Calculates the integral for bandpowers based on the composite trapezoidal rule relying on Riemann Sums.
    :param array x: array of x values
    :param array y: array of y values
    :return float: the integral of the bandpowers from the lowest l to max l
    """
    num_trapz = len(x) - 1
    widths = np.array([x[i + 1] - x[i] for i in range(num_trapz)])
    trapz_heights = np.array([y[i] + y[i + 1] for i in range(num_trapz)])
    trapz_areas = 0.5 * widths * trapz_heights
    return np.sum(trapz_areas) / (x[-1] - x[0])


def average_bands(l, cls, n):
    """
    n is the number of bands
    """
    band_cutoffs = np.logspace(np.log(min(l)), np.log(max(l) + 1), n + 1, base=np.e)

    binned_l = []
    binned_cls = []
    for i in range(n):
        temp_l = l[np.logical_and(l >= band_cutoffs[i], l < band_cutoffs[i + 1])]
        temp_cls = cls[np.logical_and(l >= band_cutoffs[i], l < band_cutoffs[i + 1])]
        binned_l.append(np.array(temp_l))
        binned_cls.append(np.array(temp_cls))

    bandpowers = np.zeros(n)
    for i in range(n):
        bandpowers[i] = bandpower_integral(binned_l[i], binned_cls[i])

    return bandpowers


def make_bandpowers(
    mock_run,
    mocks_dir,
    mocks_name,
    ell_val="shear_cl--ell",
    cl_vals=["shear_cl--bin"],
    num_bands=8,
    n_bin_pairs=15,
    l_min=1.0,
    reorder=False,
    covariance=None,
    to_write_headers=["bandpowers"],
    file_locs=["bandpowers/bandpowers.txt"],
):
    assert isinstance(cl_vals, list), "The defined cl_vals must be wrapped up in a list"
    assert len(cl_vals) == len(
        to_write_headers
    ), "Mismatch between the number of cl_vals and file headers"
    assert len(cl_vals) == len(
        file_locs
    ), "Mismatch between the number of cl_vals and file locations to write"
    vals_to_read = list(cl_vals)
    vals_to_read.append(ell_val)
    # Read the input ell.
    mock_obj = km.read_kcap_values(
        mock_run=mock_run, mocks_dir=mocks_dir, mocks_name=mocks_name
    )
    vals_dict = mock_obj.read_vals(vals_to_read=vals_to_read)
    ell = vals_dict[ell_val]
    ell = ell[ell >= l_min]  # Filter out the ell = 0 value
    num_ell = len(ell)

    bandpowers_stacked = np.zeros((len(cl_vals), n_bin_pairs * num_bands))
    for i, cl_val in enumerate(cl_vals):
        temp_bandpower = np.zeros((n_bin_pairs, num_bands))
        cl = vals_dict[cl_val].reshape(n_bin_pairs, -1)[:, -num_ell:]
        for j in range(n_bin_pairs):
            temp_bandpower[j] += average_bands(ell, cl[i], num_bands)
        bandpowers_stacked[i] = temp_bandpower.flatten()

    if reorder is True:
        for i, bandpower_stacked in enumerate(bandpowers_stacked):
            bandpowers_stacked[i] = bandpower_stacked.reshape(
                n_bin_pairs, -1
            ).T.flatten()

    if covariance is not None:
        noisey_bandpowers_stacked = bandpowers_stacked.copy()
        for i, noisey_bandpower_stacked in enumerate(noisey_bandpowers_stacked):
            noisey_bandpowers_stacked[i] = np.random.multivariate_normal(
                noisey_bandpower_stacked, covariance
            )
        bandpowers_stacked = np.vstack((bandpowers_stacked, noisey_bandpowers_stacked))

        to_write_headers.extend(["noisey_" + val for val in to_write_headers])
        file_locs.extend(
            [val.split("/")[0] + "/noisey_" + val.split("/")[1] for val in file_locs]
        )  # adds the necessary file paths and headers for the files to write
    print("Writing bandpowers")
    mock_obj.write_to_tar(
        to_write=list(bandpowers_stacked),
        to_write_headers=to_write_headers,
        file_loc=file_locs,
    )
    print("Written bandpowers")
    mock_obj.close()
