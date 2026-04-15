from typing import Union

# Allow "Run Python File" on DeepSlice/main.py in VS Code.
# When executed as a script, launch the GUI entrypoint from package context
# before evaluating package-relative imports below.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from DeepSlice.gui.app import main as _launch_gui_main

    raise SystemExit(_launch_gui_main())

from .coord_post_processing import spacing_and_indexing, angle_methods
from .read_and_write import QuickNII_functions
from .neural_network import neural_network
from .metadata import metadata_loader


class DSModel:
    def __init__(self, species, download_callback=None, log_callback=None):
        """Initialises a DeepSlice model for a given species
        :param species: the species of the brain to be processed, must be one of "mouse", "rat"
        :type species: str
        """
        self.species = species
        self.download_callback = download_callback
        self.log_callback = log_callback

        self.config, self.metadata_path = metadata_loader.load_config()
        xception_weights = metadata_loader.get_data_path(
            self.config["weight_file_paths"]["xception_imagenet"],
            self.metadata_path,
            download_callback=self.download_callback,
        )
        weights = metadata_loader.get_data_path(
            self.config["weight_file_paths"][self.species]["primary"],
            self.metadata_path,
            download_callback=self.download_callback,
        )
        self.model = neural_network.initialise_network(
            xception_weights, weights, self.species
        )

    @staticmethod
    def _parse_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return bool(value)

    def _log(self, message, callback=None):
        logger = callback if callback is not None else self.log_callback
        if logger is None:
            print(message)
            return
        logger(message)

    def predict(
        self,
        image_directory: str = None,
        ensemble: bool = None,
        section_numbers: bool = True,
        legacy_section_numbers=False,
        image_list=None,
        use_secondary_model=False,
        progress_callback=None,
        log_callback=None,
    ):
        """predicts the atlas position for a folder full of histological brain sections

        :param image_directory: the directory containing the brain sections
        :type image_directory: str
        :param ensemble: whether to use multiple models, this will default to True when available, defaults to None
        :type ensemble: bool, optional
        :param section_numbers: whether this dataset includes section numbers (as the last three digits of the filename) , defaults to True
        :type section_numbers: bool, optional
        :param legacy_section_numbers: a legacy setting which parses section numbers how old DeepSlice used to, defaults to False
        :type legacy_section_numbers: bool, optional
        """

        # We set this to false as predict is the entry point for a new brain and therefore we need to reset all values which may persist from a previous animal.
        self.bad_sections_present = False
        # Different species may or may not have an ensemble model, so we need to check for this before defaulting to True
        if ensemble is None:
            ensemble = self.config["ensemble_status"][self.species]
            ensemble = self._parse_bool(ensemble)
        if image_list:
            image_generator, width, height = neural_network.load_images_from_list(
                image_list
            )
            if image_directory:
                self._log(
                    "WARNING: image_directory is set but image_list is also set. image_directory will be ignored.",
                    callback=log_callback,
                )
        else:
            image_generator, width, height = neural_network.load_images_from_path(
                image_directory
            )
        primary_weights = metadata_loader.get_data_path(
            self.config["weight_file_paths"][self.species]["primary"],
            self.metadata_path,
            download_callback=self.download_callback,
        )

        secondary_weights = metadata_loader.get_data_path(
            self.config["weight_file_paths"][self.species]["secondary"],
            self.metadata_path,
            download_callback=self.download_callback,
        )

        if secondary_weights == "None":
            self._log(f"ensemble is not available for {self.species}", callback=log_callback)
            if use_secondary_model:
                self._log(
                    "WARNING: use_secondary_model is set but no secondary model is available. use_secondary_model will be ignored.",
                    callback=log_callback,
                )
                use_secondary_model = False
            ensemble = False
        if use_secondary_model and ensemble:
            self._log(
                "WARNING: use_secondary_model is set but ensemble is also set. use_secondary_model will be ignored.",
                callback=log_callback,
            )
            use_secondary_model = False
        if use_secondary_model:
            self._log("Using secondary model", callback=log_callback)
            predictions = neural_network.predictions_util(
                self.model,
                image_generator,
                secondary_weights,
                None,
                ensemble,
                self.species,
                progress_callback=progress_callback,
                log_callback=log_callback,
            )
        else:
            predictions = neural_network.predictions_util(
                self.model,
                image_generator,
                primary_weights,
                secondary_weights,
                ensemble,
                self.species,
                progress_callback=progress_callback,
                log_callback=log_callback,
            )
        predictions["width"] = width
        predictions["height"] = height
        if section_numbers:
            predictions["nr"] = spacing_and_indexing.number_sections(
                predictions["Filenames"], legacy_section_numbers
            )
            predictions["nr"] = predictions["nr"].astype(int)
            predictions = predictions.sort_values(by="nr").reset_index(drop=True)
        else:
            ###this is just for coronal, change later
            predictions = predictions.sort_values(by="oy").reset_index(drop=True)

        #: pd.DataFrame: Filenames and predicted QuickNII coordinates of the input sections.

        self.predictions = predictions
        self.image_directory = image_directory

    def set_bad_sections(self, bad_sections: list, auto=False):
        """
        sets the bad sections for a given brain. Must be run after predict()

        :param bad_sections: A list of bad sections to ignore when calculating angles and spacing, the list just needs to contain a unique string for each section, for instance the section number. do not use a string which appears in multiple filenames
        :type bad_sections: list
        """
        self.predictions = spacing_and_indexing.set_bad_sections_util(
            self.predictions, bad_sections, auto
        )

    def enforce_index_order(self):
        """
        reorders the section depths (oy) in the predictions such that they align with the section indexes
        """
        self.predictions = spacing_and_indexing.enforce_section_ordering(
            self.predictions
        )

    def enforce_index_spacing(
        self, section_thickness: Union[int, float] = None, suppress=False
    ):
        """
        Space evenly according to the section indexes, if these indexes do not represent the precise order in which the sections were
        cut, this will lead to less accurate predictions. Section indexes must account for missing sections (ie, if section 3 is missing
        indexes must be 1, 2, 4).

        :param section_thickness: the thickness of the sections in microns, defaults to None
        :type section_thickness: Union[int, float], optional
        """
        voxel_size = self.config["target_volumes"][self.species]["voxel_size_microns"]
        self.predictions = spacing_and_indexing.space_according_to_index(
            self.predictions,
            section_thickness=section_thickness,
            voxel_size=voxel_size,
            suppress=suppress,
            species=self.species,
        )

    def adjust_angles(self, ML: Union[int, float], DV: Union[int, float]):
        """
        Adjusts the Mediolateral (ML) and Dorsoventral (DV) angles of all sections to the specified values.

        :param ML: the Mediolateral angle to set
        :param DV: the Dorsoventral angle to set
        :type ML: [int, float]
        :type DV: [int, float]
        """
        self.predictions = angle_methods.set_angles(self.predictions, DV, ML)

    def propagate_angles(self, method="weighted_mean"):
        """
        Calculates the average Mediolateral and Dorsoventral angles for all sections.
        """
        ##needs to be run twice as adjusting the angle in one plane bumps the other out slightly.
        for i in range(2):
            self.predictions = angle_methods.propagate_angles(
                self.predictions, method, self.species
            )

    def load_QUINT(self, filename):
        """
        Load a QUINT compatible JSON or XML.

        :param filename: the name of the file to load
        :type filename: str
        """
        if filename.lower().endswith(".json"):
            predictions, target = QuickNII_functions.read_QUINT_JSON(filename)
            if target == "ABA_Mouse_CCFv3_2017_25um.cutlas" and self.species != "mouse":
                self.species = "mouse"
                print("Switching to a mouse model")
            elif target == "WHS_Rat_v4_39um.cutlas" and self.species != "rat":
                self.species = "rat"
                print("switching to a rat model")
        elif filename.lower().endswith(".xml"):
            predictions = QuickNII_functions.read_QuickNII_XML(filename)
        else:
            raise ValueError("File must be a JSON or XML")
        self.predictions = predictions
        xception_weights = metadata_loader.get_data_path(
            self.config["weight_file_paths"]["xception_imagenet"],
            self.metadata_path,
            download_callback=self.download_callback,
        )
        weights = metadata_loader.get_data_path(
            self.config["weight_file_paths"][self.species]["primary"],
            self.metadata_path,
            download_callback=self.download_callback,
        )
        self.model = neural_network.initialise_network(
            xception_weights, weights, self.species
        )

    def save_predictions(self, filename, output_format="json"):
        """
        Save the predictions to a QuickNII compatible file.

        :param filename: the name of the file to save to
        :type filename: str
        :param output_format: one of "json" or "xml", defaults to "json"
        :type output_format: str
        """
        target = self.config["target_volumes"][self.species]["name"]
        aligner = self.config["DeepSlice_version"]["prerelease"]
        self.predictions.to_csv(filename + ".csv", index=False)
        output_format = output_format.lower()
        if output_format == "json":
            QuickNII_functions.write_QUINT_JSON(
                df=self.predictions, filename=filename, aligner=aligner, target=target
            )
        elif output_format == "xml":
            QuickNII_functions.write_QuickNII_XML(
                df=self.predictions, filename=filename, aligner=aligner
            )
        else:
            raise ValueError("output_format must be one of 'json' or 'xml'")
