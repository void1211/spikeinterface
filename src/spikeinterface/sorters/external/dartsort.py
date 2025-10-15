from spikeinterface.sorters import BaseSorter
from spikeinterface.extractors import BinaryRecordingExtractor
from spikeinterface.core import NumpySorting

import os
import json
import numpy as np
# from typing import override

try:
    import dartsort
    HAVE_DARTSORT = True
except ImportError:
    HAVE_DARTSORT = False

class DARTsortSorter(BaseSorter):
    """DARTsort Sorter object."""

    sorter_name = "dartsort"
    requires_locations = True
    gpu_capability = "nvidia-required"
    requires_binary_data = True

    installation_mesg = """
        To use dartsort, please install dartsort from https://github.com/cwindolf/dartsort
    """

    @classmethod
    def _dynamic_params(cls):
        if cls.is_installed():
            try:
                # DeveloperConfigを優先（UserConfig + Developer専用パラメータを全て含む）
                try:
                    from dartsort.config import DeveloperConfig
                    default_config = DeveloperConfig()
                except (ImportError, AttributeError):
                    # フォールバック: DARTsortUserConfig
                    from dartsort.config import DARTsortUserConfig
                    default_config = DARTsortUserConfig()
                
                # pydantic dataclassのフィールド情報から動的に全パラメータを取得
                default_params = {}
                params_description = {}
                
                # __dataclass_fields__から全フィールドを取得
                # DeveloperConfigの場合、親クラス（UserConfig）のフィールドも自動的に含まれる
                if hasattr(default_config, '__dataclass_fields__'):
                    for field_name, field_info in default_config.__dataclass_fields__.items():
                        # デフォルト値を取得
                        default_value = getattr(default_config, field_name)
                        default_params[field_name] = default_value
                        
                        # ドキュメント文字列を取得
                        doc = ""
                        # metadata内のdocフィールドをチェック
                        if hasattr(field_info, 'metadata') and 'doc' in field_info.metadata:
                            doc = field_info.metadata['doc']
                        # default内のドキュメントをチェック
                        elif hasattr(field_info, 'default'):
                            if hasattr(field_info.default, '__doc__') and field_info.default.__doc__:
                                doc = field_info.default.__doc__
                            # pydantic Fieldのdescriptionをチェック
                            elif hasattr(field_info.default, 'description') and field_info.default.description:
                                doc = field_info.default.description
                        params_description[field_name] = doc
                
                return default_params, params_description
            except Exception as e:
                # エラーが発生した場合はデフォルト値を返す
                import warnings
                warnings.warn(f"Could not load DARTsort config dynamically: {e}")
                return cls._default_params, cls._params_description
        else:
            return cls._default_params, cls._params_description

    @classmethod
    def get_sorter_version(cls):
        """Get dartsort version."""
        try:
            return dartsort.__version__
        except:
            return "Unknown"
    
    @classmethod
    def get_sorter_name(cls):
        """Get dartsort name."""
        return "dartsort"

    @classmethod
    def is_installed(cls):
        """Check if dartsort is installed."""
        try:
            import dartsort
            return True
        except ImportError:
            return False

    @classmethod
    def _setup_recording(cls, recording, sorter_output_folder, params, verbose):
        """Setup recording for dartsort."""
        # レコーディングをバイナリファイルとして保存
        binary_file_path = sorter_output_folder / "recording.dat"
        from spikeinterface.core import write_binary_recording
        write_binary_recording(
            recording=recording,
            file_paths=[binary_file_path],
            dtype='float32',
            verbose=verbose
        )
        
        # レコーディング情報を保存
        recording_info = {
            'sampling_frequency': recording.get_sampling_frequency(),
            'num_channels': recording.get_num_channels(),
            'num_samples': recording.get_num_samples(),
            'channel_ids': recording.get_channel_ids().tolist()
        }
        recording_info_file = sorter_output_folder / "recording_info.json"
        with open(recording_info_file, 'w') as f:
            json.dump(recording_info, f, indent=2)
        
        # プローブ情報がある場合は保存
        if recording.get_probe() is not None:
            probe_file_path = sorter_output_folder / "probe.json"
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            probe_dict = recording.get_probe().to_dict()
            probe_dict_serializable = convert_numpy(probe_dict)
            with open(probe_file_path, 'w') as f:
                json.dump(probe_dict_serializable, f, indent=2)

    @classmethod
    def _run_from_folder(cls, output_folder, sorter_params, verbose):
        """Run dartsort from folder."""
        dat_file_path = os.path.join(output_folder, 'recording.dat')
        
        # レコーディング情報を読み込み
        recording_info_file = os.path.join(output_folder, 'recording_info.json')
        
        try:
            import dartsort
            from spikeinterface.extractors import BinaryRecordingExtractor
        except ImportError:
            raise ImportError("DARTsort package is not installed. Please install it from GitHub.")
        
        dat_file_path = os.path.join(output_folder, 'recording.dat')
        
        # レコーディング情報を読み込み
        recording_info_file = os.path.join(output_folder, 'recording_info.json')
        if not os.path.exists(recording_info_file):
            parent_dir = os.path.dirname(output_folder)
            recording_info_file = os.path.join(parent_dir, 'recording_info.json')
        
        if not os.path.exists(recording_info_file):
            raise FileNotFoundError(f"recording_info.json not found in {output_folder} or {parent_dir}")
        
        with open(recording_info_file, 'r') as f:
            recording_info = json.load(f)
        
        sampling_frequency = recording_info['sampling_frequency']
        num_channels = recording_info['num_channels']
        
        # レコーディングを再構築
        recording = BinaryRecordingExtractor(
            file_paths=[dat_file_path],
            sampling_frequency=sampling_frequency,
            num_channels=num_channels,
            dtype='float32'
        )
        
        # プローブ情報がある場合は設定
        probe_file_path = os.path.join(output_folder, 'probe.json')
        if os.path.exists(probe_file_path):
            from probeinterface import Probe
            with open(probe_file_path, 'r') as f:
                probe_dict = json.load(f)
            probe = Probe.from_dict(probe_dict)
            recording = recording.set_probe(probe)
        
        # DARTsortの設定を作成
        try:
            from dartsort.config import DARTsortUserConfig
            
            # DeveloperConfigが利用可能かチェック
            try:
                from dartsort.config import DeveloperConfig
                has_developer_config = True
            except (ImportError, AttributeError):
                has_developer_config = False
            
            # 両方の設定クラスの有効なフィールドを取得
            user_fields = set()
            developer_fields = set()
            
            # UserConfigのフィールドを取得
            user_config_tmp = DARTsortUserConfig()
            if hasattr(user_config_tmp, '__dataclass_fields__'):
                user_fields = set(user_config_tmp.__dataclass_fields__.keys())
            
            # DeveloperConfigのフィールドを取得（利用可能な場合）
            if has_developer_config:
                developer_config_tmp = DeveloperConfig()
                if hasattr(developer_config_tmp, '__dataclass_fields__'):
                    developer_fields = set(developer_config_tmp.__dataclass_fields__.keys())
            
            # 全ての有効なフィールド
            all_valid_fields = user_fields | developer_fields
            
            # パラメータを抽出
            cfg_params = {}
            uses_developer_params = False
            
            for key, value in sorter_params.items():
                # 互換性のための古いパラメータ名のマッピング
                if key == 'detect_threshold':
                    if 'initial_threshold' in all_valid_fields:
                        cfg_params['initial_threshold'] = value
                elif key == 'run_gpu':
                    if 'device' in all_valid_fields:
                        cfg_params['device'] = 'cuda' if value else 'cpu'
                # 直接のパラメータ名の場合
                elif key in all_valid_fields:
                    cfg_params[key] = value
                    # Developer専用パラメータが使われているかチェック
                    if key in developer_fields and key not in user_fields:
                        uses_developer_params = True
                # job_kwargsなどのSpikeInterface固有のパラメータは無視
                elif key not in ['n_jobs', 'chunk_duration', 'progress_bar', 'mp_context']:
                    if verbose:
                        print(f"Warning: Unknown parameter '{key}' will be ignored")
            
            # Developer専用パラメータが使われている場合はDeveloperConfigを使用
            if has_developer_config and uses_developer_params:
                cfg = DeveloperConfig(**cfg_params)
                if verbose:
                    print("Using DeveloperConfig for DARTsort")
            else:
                cfg = DARTsortUserConfig(**cfg_params)
                if verbose:
                    print("Using DARTsortUserConfig for DARTsort")
            
        except Exception as e:
            # 設定の作成に失敗した場合はデフォルト設定を使用
            cfg = None
            if verbose:
                print(f"Warning: Could not create DARTsort config: {e}")
        
        if verbose:
            print(f"Running DARTsort with parameters: {sorter_params}")
        
        # DARTsortを実行
        try:
            if cfg is not None:
                result = dartsort.dartsort(
                    recording=recording,
                    output_dir=output_folder,
                    cfg=cfg,
                    overwrite=True
                )
            else:
                result = dartsort.dartsort(
                    recording=recording,
                    output_dir=output_folder,
                    overwrite=True
                )
            
            if verbose:
                print(f"DARTsort completed successfully.")
                
        except Exception as e:
            raise RuntimeError(f"DARTsort execution failed: {e}")

    @classmethod
    def _get_result_from_folder(cls, output_folder):
        """Get result from folder."""
        print(f"Getting result from folder: {output_folder}")
        try:
            from dartsort.util.data_util import DARTsortSorting
            
            # dartsort_sorting.npzファイルから読み込み
            sorting_npz_path = os.path.join(output_folder, 'dartsort_sorting.npz')
            if os.path.exists(sorting_npz_path):
                sorting = DARTsortSorting.load(sorting_npz_path)
                return sorting.to_numpy_sorting()
            else:
                # フォールバック: HDF5ファイルから読み込み
                h5_files = [f for f in os.listdir(output_folder) if f.endswith('.h5')]
                if h5_files:
                    h5_path = os.path.join(output_folder, h5_files[0])
                    sorting = DARTsortSorting.from_peeling_hdf5(h5_path)
                    return sorting.to_numpy_sorting()
                else:
                    raise FileNotFoundError("No dartsort output files found")
            
        except ImportError:
            # DARTsortSortingが利用できない場合はフォールバック
            print(f"DARTsortSorting not available: {e}")
            pass
        except Exception as e:
            # その他のエラーもフォールバック
            print(f"Error getting result from folder: {e}")
            pass
        
        return sorting