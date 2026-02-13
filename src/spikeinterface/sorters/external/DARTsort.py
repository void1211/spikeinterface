from spikeinterface.sorters import BaseSorter
from spikeinterface.extractors import BinaryRecordingExtractor
from spikeinterface.core import NumpySorting

import os
import json
import numpy as np
# from typing import override

try:
    import dartsort
    HAVE_DARTsort = True
except ImportError:
    try:
        import DARTsort as dartsort
        HAVE_DARTsort = True
    except ImportError:
        HAVE_DARTsort = False

class DARTsortSorter(BaseSorter):
    """DARTsort Sorter object."""

    sorter_name = "dartsort"
    requires_locations = True
    gpu_capability = "nvidia-required"
    requires_binary_data = True

    _default_params = {}
    _params_description = {}

    # 動的読み込み失敗時用（h5py 未導入など）。dartsort の有効なパラメータ名を列挙し set_params_to_folder の検証を通す
    _si_default_params = {
        "gmm_max_spikes": 1024000,
        "core_radius": 10,
        "pool_engine": "process",
        "n_jobs": 1,
        "chunk_duration": "1s",
        "progress_bar": True,
        "mp_context": None,
        "max_threads_per_worker": 1,
    }

    installation_mesg = """
        To use DARTsort, please install DARTsort from https://github.com/cwindolf/dartsort
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
                    try:
                        from dartsort.config import DARTsortUserConfig
                        default_config = DARTsortUserConfig()
                    except ImportError:
                        raise
                
                # pydantic dataclassのフィールド情報から動的に全パラメータを取得
                default_params = {}
                params_description = {}
                
                # __dataclass_fields__ または model_fields (Pydantic v2) から取得
                if hasattr(default_config, '__dataclass_fields__'):
                    fields_src = default_config.__dataclass_fields__
                elif hasattr(default_config, 'model_fields'):
                    fields_src = default_config.model_fields
                else:
                    fields_src = {}
                for field_name in (fields_src.keys() if hasattr(fields_src, 'keys') else []):
                    field_info = fields_src[field_name]
                    default_params[field_name] = getattr(default_config, field_name, None)
                    doc = ""
                    if hasattr(field_info, 'metadata') and isinstance(getattr(field_info, 'metadata', None), dict) and 'doc' in field_info.metadata:
                        doc = field_info.metadata.get('doc', '')
                    elif hasattr(field_info, 'default'):
                        if hasattr(field_info.default, '__doc__') and field_info.default.__doc__:
                            doc = field_info.default.__doc__ or ""
                        elif hasattr(field_info.default, 'description') and field_info.default.description:
                            doc = getattr(field_info.default, 'description', '') or ''
                    params_description[field_name] = doc
                default_params.update(cls._si_default_params)
                return default_params, params_description
            except Exception as e:
                # パッケージメタデータのエラー（gitからインストールした場合など）は無視
                # dartsort/__init__.pyで__version__ = importlib.metadata.version("dartsort")が実行されるため
                import warnings
                error_msg = str(e)
                if "package metadata" not in error_msg.lower() and "No package metadata" not in error_msg:
                    warnings.warn(f"Could not load DARTsort config dynamically: {e}")
                return cls._si_default_params.copy(), cls._params_description
        else:
            return cls._si_default_params.copy(), cls._params_description

    @classmethod
    def get_sorter_version(cls):
        """Get DARTsort version."""
        try:
            return dartsort.__version__
        except:
            return "Unknown"
    
    @classmethod
    def get_sorter_name(cls):
        """Get DARTsort name."""
        return "dartsort"

    @classmethod
    def is_installed(cls):
        """Check if DARTsort is installed."""
        import sys
        import os
        import importlib.util
        from pathlib import Path
        
        # まず、モジュールが存在するかどうかを確認（実際にインポートしない）
        # これにより、AssertionErrorなどのインポート時のエラーを回避
        dartsort_spec = importlib.util.find_spec("dartsort")
        if dartsort_spec is not None:
            # モジュールが見つかった場合、実際にインポートを試みる
            # ただし、AssertionErrorなどのエラーは無視する
            try:
                import dartsort
                return True
            except (ImportError, AssertionError, AttributeError):
                # インポートエラーやアサーションエラーは無視
                # モジュールが存在する場合はTrueを返す（実行時のエラーは別途処理）
                return True
            except Exception:
                # その他のエラーも無視（依存関係の問題など）
                return True
        
        # DARTsortという名前でも確認
        DARTsort_spec = importlib.util.find_spec("DARTsort")
        if DARTsort_spec is not None:
            try:
                import DARTsort
                return True
            except (ImportError, AssertionError, AttributeError):
                return True
            except Exception:
                return True
        
        # sys.pathに追加する可能性のあるパスをチェック
        # 環境変数から取得、または一般的なパスをチェック
        possible_paths = []
        
        # 環境変数から取得
        if 'DARTsort_PATH' in os.environ:
            possible_paths.append(Path(os.environ['DARTsort_PATH']) / "src" / "dartsort")
        
        # 一般的なパスをチェック
        possible_paths.extend([
            Path("C:/Users/tanaka-users/tlab/yasui/dartsort/src/dartsort"),
            Path("C:/Users/tanaka-users/tlab/yasui/dartsort2/src/dartsort"),
            Path.home() / "dartsort" / "src" / "dartsort",
        ])
        
        # パスが存在する場合はsys.pathに追加して再試行
        for dartsort_path in possible_paths:
            if dartsort_path.exists():
                parent_path = dartsort_path.parent
                if str(parent_path) not in sys.path:
                    sys.path.insert(0, str(parent_path))
                    # モジュールが存在するか確認
                    spec = importlib.util.find_spec("dartsort")
                    if spec is not None:
                        return True
        
        return False

    @classmethod
    def _setup_recording(cls, recording, sorter_output_folder, params, verbose):
        """Setup recording for DARTsort."""
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
    def _run_from_folder(cls, sorter_output_folder, sorter_params, verbose):
        """Run DARTsort from folder.
        
        Parameters
        ----------
        sorter_output_folder : Path
            The sorter output folder (output_folder / "sorter_output")
        sorter_params : dict
            The sorter parameters
        verbose : bool
            Whether to print verbose output
        """
        dat_file_path = os.path.join(sorter_output_folder, 'recording.dat')
        
        # レコーディング情報を読み込み
        recording_info_file = os.path.join(sorter_output_folder, 'recording_info.json')
        
        try:
            import dartsort
        except ImportError as e:
            # より詳細なエラーメッセージを提供
            error_msg = str(e)
            if "dredge" in error_msg.lower():
                raise ImportError(
                    "dartsort package is installed but missing required dependency 'dredge'. "
                    "Please install it with: pip install dredge "
                    "or install all DARTsort dependencies with: pip install -r requirements-full.txt"
                ) from e
            raise ImportError(
                    f"dartsort package is not installed or has missing dependencies. "
                    f"Original error: {error_msg}. "
                    f"Please install DARTsort from https://github.com/cwindolf/dartsort"
                ) from e
        except Exception as e:
            # その他のエラー（AssertionErrorなど）もキャッチ
            raise ImportError(
                f"Failed to import dartsort: {e}. "
                f"Please ensure DARTsort and all its dependencies are installed correctly."
            ) from e
        
        from spikeinterface.extractors import BinaryRecordingExtractor
        if not os.path.exists(recording_info_file):
            parent_dir = os.path.dirname(sorter_output_folder)
            recording_info_file = os.path.join(parent_dir, 'recording_info.json')
        
        if not os.path.exists(recording_info_file):
            raise FileNotFoundError(f"recording_info.json not found in {sorter_output_folder} or {parent_dir}")
        
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
        probe_file_path = os.path.join(sorter_output_folder, 'probe.json')
        if os.path.exists(probe_file_path):
            from probeinterface import Probe
            with open(probe_file_path, 'r') as f:
                probe_dict = json.load(f)
            probe = Probe.from_dict(probe_dict)
            recording = recording.set_probe(probe)
        
        # DARTsortの設定を作成
        try:
            from dartsort.config import DARTsortUserConfig
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
            try:
                if cfg is not None:
                    result = dartsort.dartsort(
                        recording=recording,
                        output_dir=sorter_output_folder,
                        cfg=cfg,
                        overwrite=True
                    )
                else:
                    result = dartsort.dartsort(
                        recording=recording,
                        output_dir=sorter_output_folder,
                        overwrite=True
                    )
            except Exception as e:
                raise RuntimeError(f"DARTsort execution failed: {e}")
            if verbose:
                print(f"DARTsort completed successfully.")
                
        except Exception as e:
            raise RuntimeError(f"DARTsort execution failed: {e}")

    @classmethod
    def _get_result_from_folder(cls, sorter_output_folder):
        """Get result from folder.
        
        Parameters
        ----------
        sorter_output_folder : Path
            The sorter output folder (output_folder / "sorter_output")
        """
        import warnings
        if not os.path.exists(sorter_output_folder):
            raise FileNotFoundError(f"Output folder not found: {sorter_output_folder}")
        
        try:
            from dartsort.util.data_util import DARTsortSorting
            
            # DARTsort_sorting.npzファイルから読み込み
            sorting_npz_path = os.path.join(sorter_output_folder, 'DARTsort_sorting.npz')
            if os.path.exists(sorting_npz_path):
                sorting = DARTsortSorting.load(sorting_npz_path)
                return sorting.to_numpy_sorting()
            else:
                # フォールバック: HDF5ファイルから読み込み
                h5_files = [f for f in os.listdir(sorter_output_folder) if f.endswith('.h5')]
                if h5_files:
                    h5_path = os.path.join(sorter_output_folder, h5_files[0])
                    sorting = DARTsortSorting.from_peeling_hdf5(h5_path)
                    return sorting.to_numpy_sorting()
                else:
                    raise FileNotFoundError(
                        f"No DARTsort output files found in {sorter_output_folder}. "
                        f"Expected 'DARTsort_sorting.npz' or '.h5' files."
                    )
            
        except ImportError as e:
            raise ImportError(f"DARTsortSorting not available: {e}")
        except Exception as e:
            raise RuntimeError(f"Error getting result from folder: {e}")