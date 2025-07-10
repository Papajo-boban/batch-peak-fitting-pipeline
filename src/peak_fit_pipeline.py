"""
Author: Tomáš Schuster
Affiliation: VU Amsterdam, 2025


Performs batch peak fitting (Gaussian or Pseudo-Voigt) on 2D mesh spectral data in HDF5 files,
using a YAML configuration and parallel processing.

See README.md for full usage, configuration, and output details.
"""
import os
import yaml
import h5py
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
# Uses non-interactive 'Agg' backend to enable saving fit diagnostic plots without requiring a display
from matplotlib import pyplot as plt
from lmfit import Model
from lmfit.models import PseudoVoigtModel
from joblib import Parallel, delayed
import warnings
import plotly.graph_objects as go


GAUSSIAN_FIT_PARAM_TAGS = ['cen','amp','wid','const','slope','reduced_chi_sqr','peak_area']
PSEUDOVOIGT_FIT_PARAM_TAGS = ['center','amplitude','sigma','fraction','reduced_chi_sqr','peak_area']
"""param tags are specified by the lmfit"""
ALLOWED_MODELS = ['gaussian', 'pseudo_voigt']
CHUNK_SIZE = 3000
"""CHUNK_SIZE defines how many mesh points to process per parallel batch.
Adjust based on available system RAM and CPU cores.
"""



class MultiFittingConfig:
    """
    Loads and manages multiple fitting configurations from a YAML file.
    """
    def __init__(self, config_file="config.yaml"):
        with open(config_file, 'r') as f:
            full_config = yaml.safe_load(f)
            self.configs = [FittingConfig(cfg) for cfg in full_config['datasets']]


class FittingConfig:
    """
    Parses and validates a single dataset configuration for peak fitting.
    """
    def __init__(self, config: dict):
        self.input_path = config['input_path']
        base_name = os.path.basename(self.input_path)      
        self.dataset_name = os.path.splitext(base_name)[0]
        self.output_path = config['output_path']
        self.model = config['fitting_model']
        self.roi_list = config['roi_list']
        
        self.aspect_ratio = config.get('aspect_ratio', 1.0)
        numerator, denominator = map(float, self.aspect_ratio.split('/'))
        self.aspect_ratio = numerator / denominator

        self.save_diagnostics = config['save_diagnostics']
        self.diagnostic_point_index = config['diagnostic_point_index']
        self.save_scatter_plot = config['save_interactive_scattering_plot']
        self.extra_clean_param_maps = config['save_extra_clean_param_maps']

        self.check_file_is_free()
        self.calc_roi_ranges()
        self.check_config()

            
    def calc_roi_ranges(self):
        # Computes the q-range around each ROI peak
        for roi in self.roi_list:
            roi["range"] = [round(roi["peak"] - roi["half_width"],4), round(roi["peak"] + roi["half_width"],4)]
            print(f"ROI range for {roi['name']}: {roi['range']}")

    def check_file_is_free(self):
        try:
            with h5py.File(self.input_path, 'r+') as self.f:
                print(f"Successfully opened '{self.input_path}'.\n")

        except OSError as e:
            if "Unable to open file" in str(e).lower() or "unable to synchronously open file" in str(e).lower():
                print(f"Error: Cannot access '{self.input_path}'. It may already be open or locked by another process.")
                sys.exit(1)
            else:
                raise  # re-raises unexpected OSError
    
    def check_config(self):
        """
        Validates the dataset configuration to ensure all fields are present and correctly formatted.
        Raises informative errors where issues are detected.
        """

        # input_path
        if not isinstance(self.input_path, str) or not self.input_path.endswith('.h5'):
            raise ValueError(f"Invalid input_path: must be a string ending in .h5. Got: {self.input_path}")
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input path {self.input_path} does not exist.")

        # output_path
        if not isinstance(self.output_path, str):
            raise ValueError("output_path must be a string.")
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            print(f"Created output directory: {self.output_path}")

        # fitting_model
        if self.model not in ALLOWED_MODELS:
            raise ValueError(f"Invalid fitting_model '{self.model}'. Allowed models are: {ALLOWED_MODELS}.")

        # aspect_ratio
        if isinstance(self.aspect_ratio, str):
            try:
                numerator, denominator = map(float, self.aspect_ratio.split("/"))
                if numerator <= 0 or denominator <= 0:
                    raise ValueError
            except Exception:
                raise ValueError(f"aspect_ratio must be a valid fraction like '4/3'. Got: {self.aspect_ratio}")
        elif isinstance(self.aspect_ratio, (int, float)):
            if self.aspect_ratio <= 0:
                raise ValueError(f"aspect_ratio must be a positive number. Got: {self.aspect_ratio}")
        else:
            raise ValueError("aspect_ratio must be a string (e.g. '4/3') or a positive number.")

        # roi_list
        if not isinstance(self.roi_list, list) or len(self.roi_list) == 0:
            raise ValueError("roi_list must be a non-empty list.")
        for i, roi in enumerate(self.roi_list):
            if "name" not in roi or not isinstance(roi["name"], str) or not roi["name"].strip():
                raise ValueError(f"ROI at index {i} has invalid or missing 'name'.")
            if "peak" not in roi or not isinstance(roi["peak"], (int, float)):
                raise ValueError(f"ROI '{roi.get('name', f'index {i}')}' has invalid or missing 'peak' (must be numeric).")
            if roi["peak"] > 54 or roi["peak"] < 1:
                raise ValueError(f"ROI '{roi['name']}' has 'peak' value out of valid range (1 to 54).")
            if "half_width" not in roi or not isinstance(roi["half_width"], (int, float)) or roi["half_width"] <= 0:
                raise ValueError(f"ROI '{roi['name']}' has invalid or missing 'half_width' (must be positive number).")
        
        # save_diagnostics
        if not isinstance(self.save_diagnostics, bool):
            raise ValueError("save_diagnostics must be a boolean (True or False).")

        # single_point_index
        if not isinstance(self.diagnostic_point_index, int) or self.diagnostic_point_index < 0:
            raise ValueError(f"single_point_index must be a non-negative integer. Got: {self.diagnostic_point_index}")

        # visuals
        if not isinstance(self.save_scatter_plot, bool):
            raise ValueError("visuals must be a boolean (True or False).")

        # extra_pure_image
        if not isinstance(self.extra_clean_param_maps, bool):
            raise ValueError("extra_pure_image must be a boolean (True or False).")


class PeakFitFunctions:
    """
    Contains reusable fitting methods for both single-point and full-mesh operations,
    including model initialization, diagnostics, and image output.
    """
    def init_gaussian_model(self, roi_list, j):
        """
        Initializes a Gaussian model and its parameters for a given ROI
        """
        def gaussian (x,amp,cen,wid,const,slope):
            return (slope*const)+(amp *np.exp(-(x-cen)**2/wid))
        model=Model(gaussian)
        model.set_param_hint('cen',value = np.mean((roi_list[j]["range"][0],roi_list[j]["range"][1])),min=0)
        model.set_param_hint('amp',value = 1,min=0)
        model.set_param_hint('wid',value = 0.1,min=0)
        model.set_param_hint('const',value = 0,min=0)
        model.set_param_hint('slope',value = 1)
        params = model.make_params()
        return model, params

    def fit_single_point_model(self, i, azim_avg_intensity, q_roi, model, params, model_type, num_points, mesh_shape=None, fit_h5_list=None,roi_list=None, output_path=None, roi_index=None, save_diagnostics=False):
        """
        Fits a model (Gaussian or Pseudo-Voigt) to a single mesh point.
        Returns: tuple(index, result_values)
        """
        # Suppress warning for zero standard deviation in flat regions
        warnings.filterwarnings("ignore", message=".*std_dev==0.*")
        self.print_progress(i, num_points)
        intensity_roi = azim_avg_intensity[i, :]

        fit_tags = GAUSSIAN_FIT_PARAM_TAGS if model_type == "gaussian" else PSEUDOVOIGT_FIT_PARAM_TAGS
        if model_type == "gaussian":
            result = model.fit(intensity_roi, x=q_roi, params=params, max_nfev=1000)
        elif model_type == "pseudo_voigt":
            guessed_params = model.guess(intensity_roi, x=q_roi)
            result = model.fit(intensity_roi, x=q_roi, params=guessed_params, max_nfev=1000)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Prepare for optional HDF5 writing
        if mesh_shape and fit_h5_list and roi_index is not None:
            y, x = divmod(i, mesh_shape[1])
        else:
            y = x = None
        
        # Extract values and optionally write
        values = []
        for param_idx, tag in enumerate(fit_tags):
            if tag == "chi_squared" or tag == "reduced_chi_sqr":
                dof = result.ndata - result.nvarys
                value = result.chisqr / dof if dof > 0 else np.nan
            elif tag == "peak_area":
                value = np.trapezoid(result.best_fit, x=q_roi)
            else:
                value = result.best_values.get(tag, np.nan)

            values.append(value)

            if y is not None and x is not None and fit_h5_list and fit_h5_list[roi_index] is not None:
                fit_h5_list[roi_index][param_idx, y, x] = value

        if save_diagnostics:
            self.plot_fit_diagnostics(
                model_type,
                q_roi=q_roi,
                raw_data=intensity_roi,
                fit_result=result,
                point_idx=i,
                roi_name=roi_list[roi_index]['name'] if roi_list is not None else f"roi_{roi_index}",
                save_path=os.path.join(output_path, f"diagnostic_point_{i}_{roi_list[roi_index]['name']}.png") if output_path is not None and roi_list is not None and roi_list[roi_index]['name'] is not None else None
            )

        return i, tuple(values)
    
    def init_model(self, model_type: str, roi_list, roi_idx):
        """
        Initializes model and parameter guesses based on type.
        Returns a tuple: (model_instance, initial_params or None)
        """
        if model_type == "gaussian":
            return self.init_gaussian_model(roi_list, roi_idx)
        elif model_type == "pseudo_voigt":
            model = PseudoVoigtModel()
            return model, None  # We'll use model.guess() later
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def perform_model(self, model_type: str,model,
            azim_avg_intensity, q_roi, fit_h5_list, j, mesh_shape,
            roi_list, dataset_name, scan_no, output_path,
            params, num_points, extra_clean_param_maps, num_chunks):
        """
        Generic peak fitting loop for both Gaussian and Pseudo-Voigt models.
        """
        fit_tags = GAUSSIAN_FIT_PARAM_TAGS if model_type == "gaussian" else PSEUDOVOIGT_FIT_PARAM_TAGS

        for chunk_idx in range(num_chunks):
            start = chunk_idx * CHUNK_SIZE
            end = min((chunk_idx + 1) * CHUNK_SIZE, num_points)

            tasks = [
                delayed(self.fit_single_point_model)(
                    i=i,
                    azim_avg_intensity=azim_avg_intensity,
                    q_roi=q_roi,
                    model=model,
                    params=params,
                    model_type=model_type,
                    num_points=num_points,
                )
                for i in range(start, end)
            ]
            # Runs the fitting tasks in parallel using 60 processes.
            # Note: The optimal number of jobs depends on the system.
            # On Windows, 60 is the max number of processes
            results_parallel = list(Parallel(n_jobs=60)(tasks))
            self.map_results_to_grid(fit_tags, results_parallel, start, j, mesh_shape, fit_h5_list)
            self.plotting_images(
                fit_param_tags=fit_tags,
                j=j,
                fit_h5_list=fit_h5_list,
                dataset_name=dataset_name,
                scan_no=scan_no,
                output_path=output_path,
                roi_list=roi_list,
                extra_clean_param_maps=extra_clean_param_maps,
                model_type = model_type
            )

    def plot_fit_diagnostics(self,model_type,q_roi, raw_data, fit_result, point_idx, roi_name, save_path=None):
        """
        Plot raw data, fit, and residuals for a single mesh point.

        Parameters:
        - q_roi: np.array of q-values used in fit
        - raw_data: 1D np.array of intensity values
        - fit_result: lmfit.ModelResult (returned by fit)
        - point_idx: int, global mesh index (e.g. 4677)
        - roi_name: str, name of ROI (e.g. 'EXP_005')
        - save_path: str or None, if set, will save the figure to this path
        """
        residuals = raw_data - fit_result.best_fit
        chi_sqr = fit_result.chisqr
        """Tries to retrieve the amplitude, center, and width of the peak from the fit result.
        These keys may differ depending on the model used (e.g. amp vs. amplitude) so it falls back if the first one is missing."""
        amp = fit_result.best_values.get('amp', fit_result.best_values.get('amplitude', np.nan))
        cen = fit_result.best_values.get('cen', fit_result.best_values.get('center', np.nan))
        wid = fit_result.best_values.get('wid', fit_result.best_values.get('sigma', np.nan))

        plt.figure(figsize=(10, 6))

        # Fit Plot
        plt.subplot(2, 1, 1)
        plt.plot(q_roi, raw_data, 'bo', label='Raw Data')
        plt.plot(q_roi, fit_result.best_fit, 'r-', label=f'{model_type} Fit')
        plt.title(f"Point {point_idx} – {roi_name}\nAMP={amp:.2f}, CEN={cen:.3f}, WID={wid:.3f}, χ²={chi_sqr:.2e}")
        plt.xlabel('q (Å⁻¹)')
        plt.ylabel('Intensity')
        plt.legend()
        plt.grid(True)

        # Residuals Plot
        plt.subplot(2, 1, 2)
        plt.plot(q_roi, residuals, 'k.-', label='Residuals')
        plt.axhline(0, color='gray', linestyle='--')
        plt.xlabel('q (Å⁻¹)')
        plt.ylabel('Residual')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved diagnostic plot for point {point_idx} to: {save_path}")
        else:
            plt.show()

        plt.close()

    def map_results_to_grid(self, fit_param_tags, results_parallel, start, j, mesh_shape, fit_h5_list):
        """
        Writes fitted parameter values from parallel results into the corresponding (y, x) positions of the 2D parameter maps. 
        Uses the global mesh index to correctly locate each point within the mesh grid.
        """
        for result_idx, result in enumerate(results_parallel):
            global_idx = start + result_idx
            for param_idx in range(len(fit_param_tags)):
                y, x = divmod(global_idx, mesh_shape[1])
                fit_h5_list[j][param_idx, y, x] = result[1][param_idx]

    def plotting_images(self, fit_param_tags, j, fit_h5_list,
                        dataset_name, scan_no, output_path, roi_list,extra_clean_param_maps,model_type):
        for k in range(len(fit_param_tags)):
            param_data = fit_h5_list[j][k,:,:]
            plt_title = "%s_%s_%s_%s_%s"%(dataset_name,scan_no,roi_list[j]["name"],model_type,fit_param_tags[k])
            mini = np.percentile(param_data,5)
            maxi = np.percentile(param_data,95)
            self.save_figure(param_data, plt_title, output_path, mini, maxi,extra_clean_param_maps)

    def save_figure(self, param_data, plt_title, output_path, mini, maxi,extra_clean_param_maps):
        # Full version
        plt_fig1_savename = (f"{output_path}/{plt_title}.png")
        fig, ax = plt.subplots(figsize=(30, 6))
        cax = ax.matshow(param_data, vmin=mini, vmax=maxi)
        fig.colorbar(cax, ax=ax)
        plt.title(plt_title)
        plt.subplots_adjust(right=0.85)  # Make room for colorbar
        plt.savefig(plt_fig1_savename)
        plt.close()
        
        if extra_clean_param_maps:
            # ExtraPure image
            plt_fig2_savename = (f"{output_path}/{plt_title}_pure.png")
            fig2, ax2 = plt.subplots(figsize=(30, 6))
            ax2.imshow(param_data, vmin=mini, vmax=maxi, aspect='auto')
            ax2.axis('off')  # Hide axes, ticks, and borders
            plt.savefig(plt_fig2_savename, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close(fig2)

    def print_progress(self, i, total, bar_parts=25, last_shown=[-1]):
        """
        Prints a progress bar.
        Only prints if progress increases by 4% steps.
        """
        progress_percent = int((i / total) * 100)
        progress_step = progress_percent // 4
        if i == total - 1:
            progress_percent = 100
            filled = bar_parts
            bar = "#" * filled
            print(f"\r[{bar}] {progress_percent}% ({total}/{total})", end="", flush=True)
            print()  # move to next line
            return

        if progress_step > last_shown[0]:
            last_shown[0] = progress_step
            filled = int(progress_step * (bar_parts / 25))
            bar = "#" * filled + "." * (bar_parts - filled)
            print(f"\r[{bar}] {progress_percent}% ({i}/{total})", end="", flush=True)


class PeakFitExecutor:
    """
    Executes the peak fitting process for a single dataset
    """
    def __init__(self, config: FittingConfig, peak_functions: PeakFitFunctions):
        self.cfg = config
        self.pf= peak_functions
    
    def run(self):
        self.path = self.cfg.input_path
        self.get_scan_num()
        self.get_inputs()
        self.mesh_shape = self.calc_mesh_shape()
        self.roi_list = self.cfg.roi_list
        self.output_file_name = (f"{self.cfg.output_path}/{self.cfg.dataset_name}_scan_{self.scan_no}_fit_gaussian.h5")

        self.output_file = h5py.File(self.output_file_name,'w')
        self.init_fit_dataset()

        if self.cfg.save_scatter_plot: 
            self.show_visuals()

        else: self.process_all_rois_for_scan()

        self.cfg.f.close()
        self.output_file.close()

    def show_visuals(self):
        """
        Save an interactive Plotly plot of azimuthally integrated diffraction pattern (intensity vs q).
        """
        matplotlib.use('TkAgg') #able to show plots in a window
        from matplotlib import pyplot as plt
        print("Visuals...")

        intensity_sum = np.zeros_like(self.q)
        num_rows = self.intensity.shape[0]
        for start in range(0, num_rows, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, num_rows)
            print(f"Processing rows {start} to {end}")
            intensity_np = np.array(self.intensity)
            data_chunk = np.array(intensity_np[start:end, :, :])
            chunk_sum = np.sum(data_chunk, axis=(0, 1))
            intensity_sum += chunk_sum

        intensity_avg = intensity_sum / (intensity_np.shape[0] * intensity_np.shape[1])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.q, y=intensity_avg, mode='lines+markers', name='Average Intensity'))

        fig.update_layout(
            title="Azimuthally Integrated Diffraction Pattern",
            xaxis_title="q (Å⁻¹)",
            yaxis_title="Intensity",
            yaxis_type="log",
            hovermode="x unified",
            template="plotly_white"
        )

        html_output_path = os.path.join(self.cfg.output_path, f"{self.cfg.dataset_name}_scan{self.scan_no}_azimuthal_avg.html")
        fig.write_html(html_output_path)
        print(f"Saved interactive diffraction pattern to: {html_output_path}")
    
    def get_scan_num(self):
        with h5py.File(self.cfg.input_path, 'r') as f:
            scan_groups = [key for key in f.keys() if key.replace('.', '').isdigit()]
            if not scan_groups:
                raise ValueError("No scan groups found in file.")
            scan_groups.sort(key=lambda x: float(x))
            self.scan_no = scan_groups[0]

    def get_inputs(self):
        f =h5py.File(self.path,'r+')
        """
        At this point, intensity is still not converted to np.array()
        to avoid overloading RAM when working with large datasets.
        """
        self.intensity = f[f"/{self.scan_no}/eiger_integrate/integrated/intensity"]
        self.num_points = self.intensity.shape[0] 
        
        self.q = np.array(f[f"/{self.scan_no}/eiger_integrate/integrated/q"])

        self.azim_units = f[f"/{self.scan_no}/eiger_integrate/integrated/chi"]
    
    def calc_mesh_shape(self):
        width = round((self.num_points / self.cfg.aspect_ratio) ** 0.5)
        height = int(self.num_points / width)

        # Adjust height if necessary to fit num_points
        while height * width < self.num_points:
            height += 1
        return (height, width)

    def init_fit_dataset(self):
        self.fit_h5_list = []
        for i in range (0,len(self.roi_list)):
            """
            Because q's are not evenly spaced, we need to find the indices of the closest q values to our ROI ranges
            """
            start_idx = np.argmin(np.abs(self.q[:] - self.roi_list[i]["range"][0]))
            end_idx = np.argmin(np.abs(self.q[:] - self.roi_list[i]["range"][1]))
            dset_name = (f"peak_{self.roi_list[i]["name"]}")
            fit_dataset_name = (f"peak_fit_{self.roi_list[i]["name"]}_{self.q[start_idx]}_{self.q[end_idx]}")
            dset_name = self.output_file.create_dataset(fit_dataset_name,(7,self.mesh_shape[0],self.mesh_shape[1]))
            self.fit_h5_list.append(dset_name)
    
    def get_roi_range(self,j):
        self.start_idx = np.argmin(np.abs(self.q[:] - self.roi_list[j]["range"][0]))
        self.end_idx = np.argmin(np.abs(self.q[:] - self.roi_list[j]["range"][1]))
        self.q_roi = self.q[self.start_idx:self.end_idx]

    def process_all_rois_for_scan(self):
        for j in range (0,len(self.roi_list)):         
            self.get_roi_range(j)
            print("Preparing data for fitting...\n")
            
            intensity_np = np.array(self.intensity)
            azim_avg_intensity = np.add.reduce(intensity_np[:, :, self.start_idx:self.end_idx], axis=1) / intensity_np.shape[1]
            
            model, params = self.pf.init_model(self.cfg.model, self.roi_list, j)

            if self.cfg.save_diagnostics:
                print(f"Running single-point mode at index {self.cfg.diagnostic_point_index} for ROI '{self.roi_list[j]['name']}'")

                self.pf.fit_single_point_model(
                    i=self.cfg.diagnostic_point_index,
                    azim_avg_intensity=azim_avg_intensity,
                    q_roi=self.q_roi,
                    model=model,
                    params=params,
                    model_type=self.cfg.model,
                    num_points=self.num_points,
                    mesh_shape=self.mesh_shape,
                    fit_h5_list=self.fit_h5_list,
                    roi_list=self.roi_list,
                    output_path=self.cfg.output_path,
                    roi_index=j,
                    save_diagnostics=self.cfg.save_diagnostics
                )

            else:
                num_chunks = (self.num_points + CHUNK_SIZE - 1) // CHUNK_SIZE
                print(f"Number of chunks: {num_chunks} (chunk size: {CHUNK_SIZE} points)")
                print(f"Processing scan {self.scan_no} with mesh shape: {self.mesh_shape} and {self.num_points} mesh points, each with {len(self.q_roi)} q points")

                self.pf.perform_model(
                    model_type=self.cfg.model,
                    model=model,
                    azim_avg_intensity=azim_avg_intensity,
                    q_roi=self.q_roi,
                    fit_h5_list=self.fit_h5_list,
                    j=j,
                    mesh_shape=self.mesh_shape,
                    roi_list=self.roi_list,
                    dataset_name=self.cfg.dataset_name,
                    scan_no=self.scan_no,
                    output_path=self.cfg.output_path,
                    params=params,
                    num_points=self.num_points,
                    extra_clean_param_maps=self.cfg.extra_clean_param_maps,
                    num_chunks=num_chunks
                )


if __name__ == "__main__":
    multi_config = MultiFittingConfig("config.yaml")
    print("-"*46)
    print("Number of datasets to process:", len(multi_config.configs))
    for idx, config in enumerate(multi_config.configs, start=1):
        print(f"Processing dataset {idx}/{len(multi_config.configs)}")
        peak_functions = PeakFitFunctions()
        fitter = PeakFitExecutor(config, peak_functions)
        fitter.run()



