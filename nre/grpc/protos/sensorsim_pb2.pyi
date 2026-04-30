from typing import ClassVar as _ClassVar
from typing import Iterable as _Iterable
from typing import Mapping as _Mapping
from typing import Optional as _Optional
from typing import Union as _Union

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper

from nre.grpc.protos import common_pb2 as _common_pb2

DESCRIPTOR: _descriptor.FileDescriptor

class ShutterType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UNKNOWN: _ClassVar[ShutterType]
    ROLLING_TOP_TO_BOTTOM: _ClassVar[ShutterType]
    ROLLING_LEFT_TO_RIGHT: _ClassVar[ShutterType]
    ROLLING_BOTTOM_TO_TOP: _ClassVar[ShutterType]
    ROLLING_RIGHT_TO_LEFT: _ClassVar[ShutterType]
    GLOBAL: _ClassVar[ShutterType]

class ImageFormat(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UNDEFINED: _ClassVar[ImageFormat]
    PNG: _ClassVar[ImageFormat]
    JPEG: _ClassVar[ImageFormat]
    JPEG2000: _ClassVar[ImageFormat]
    RGB_UINT8_PLANAR: _ClassVar[ImageFormat]
    AVC: _ClassVar[ImageFormat]
    AV1: _ClassVar[ImageFormat]

class LidarDeviceType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PANDAR128: _ClassVar[LidarDeviceType]
    AT128: _ClassVar[LidarDeviceType]

UNKNOWN: ShutterType
ROLLING_TOP_TO_BOTTOM: ShutterType
ROLLING_LEFT_TO_RIGHT: ShutterType
ROLLING_BOTTOM_TO_TOP: ShutterType
ROLLING_RIGHT_TO_LEFT: ShutterType
GLOBAL: ShutterType
UNDEFINED: ImageFormat
PNG: ImageFormat
JPEG: ImageFormat
JPEG2000: ImageFormat
RGB_UINT8_PLANAR: ImageFormat
AVC: ImageFormat
AV1: ImageFormat
PANDAR128: LidarDeviceType
AT128: LidarDeviceType

class EgoMaskId(_message.Message):
    __slots__ = ("camera_logical_id", "rig_config_id")
    CAMERA_LOGICAL_ID_FIELD_NUMBER: _ClassVar[int]
    RIG_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    camera_logical_id: str
    rig_config_id: str
    def __init__(self, camera_logical_id: _Optional[str] = ..., rig_config_id: _Optional[str] = ...) -> None: ...

class AvailableEgoMasksReturn(_message.Message):
    __slots__ = ("ego_mask_metadata",)
    class EgoMaskMetadata(_message.Message):
        __slots__ = ("ego_mask_id",)
        EGO_MASK_ID_FIELD_NUMBER: _ClassVar[int]
        ego_mask_id: EgoMaskId
        def __init__(self, ego_mask_id: _Optional[_Union[EgoMaskId, _Mapping]] = ...) -> None: ...

    EGO_MASK_METADATA_FIELD_NUMBER: _ClassVar[int]
    ego_mask_metadata: _containers.RepeatedCompositeFieldContainer[AvailableEgoMasksReturn.EgoMaskMetadata]
    def __init__(
        self, ego_mask_metadata: _Optional[_Iterable[_Union[AvailableEgoMasksReturn.EgoMaskMetadata, _Mapping]]] = ...
    ) -> None: ...

class LinearCde(_message.Message):
    __slots__ = ("linear_c", "linear_d", "linear_e")
    LINEAR_C_FIELD_NUMBER: _ClassVar[int]
    LINEAR_D_FIELD_NUMBER: _ClassVar[int]
    LINEAR_E_FIELD_NUMBER: _ClassVar[int]
    linear_c: float
    linear_d: float
    linear_e: float
    def __init__(
        self, linear_c: _Optional[float] = ..., linear_d: _Optional[float] = ..., linear_e: _Optional[float] = ...
    ) -> None: ...

class FthetaCameraParam(_message.Message):
    __slots__ = (
        "principal_point_x",
        "principal_point_y",
        "reference_poly",
        "pixeldist_to_angle_poly",
        "angle_to_pixeldist_poly",
        "max_angle",
        "linear_cde",
    )
    class PolynomialType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        UNKNOWN: _ClassVar[FthetaCameraParam.PolynomialType]
        PIXELDIST_TO_ANGLE: _ClassVar[FthetaCameraParam.PolynomialType]
        ANGLE_TO_PIXELDIST: _ClassVar[FthetaCameraParam.PolynomialType]

    UNKNOWN: FthetaCameraParam.PolynomialType
    PIXELDIST_TO_ANGLE: FthetaCameraParam.PolynomialType
    ANGLE_TO_PIXELDIST: FthetaCameraParam.PolynomialType
    PRINCIPAL_POINT_X_FIELD_NUMBER: _ClassVar[int]
    PRINCIPAL_POINT_Y_FIELD_NUMBER: _ClassVar[int]
    REFERENCE_POLY_FIELD_NUMBER: _ClassVar[int]
    PIXELDIST_TO_ANGLE_POLY_FIELD_NUMBER: _ClassVar[int]
    ANGLE_TO_PIXELDIST_POLY_FIELD_NUMBER: _ClassVar[int]
    MAX_ANGLE_FIELD_NUMBER: _ClassVar[int]
    LINEAR_CDE_FIELD_NUMBER: _ClassVar[int]
    principal_point_x: float
    principal_point_y: float
    reference_poly: FthetaCameraParam.PolynomialType
    pixeldist_to_angle_poly: _containers.RepeatedScalarFieldContainer[float]
    angle_to_pixeldist_poly: _containers.RepeatedScalarFieldContainer[float]
    max_angle: float
    linear_cde: LinearCde
    def __init__(
        self,
        principal_point_x: _Optional[float] = ...,
        principal_point_y: _Optional[float] = ...,
        reference_poly: _Optional[_Union[FthetaCameraParam.PolynomialType, str]] = ...,
        pixeldist_to_angle_poly: _Optional[_Iterable[float]] = ...,
        angle_to_pixeldist_poly: _Optional[_Iterable[float]] = ...,
        max_angle: _Optional[float] = ...,
        linear_cde: _Optional[_Union[LinearCde, _Mapping]] = ...,
    ) -> None: ...

class OpenCVPinholeCameraParam(_message.Message):
    __slots__ = (
        "principal_point_x",
        "principal_point_y",
        "focal_length_x",
        "focal_length_y",
        "radial_coeffs",
        "tangential_coeffs",
        "thin_prism_coeffs",
    )
    PRINCIPAL_POINT_X_FIELD_NUMBER: _ClassVar[int]
    PRINCIPAL_POINT_Y_FIELD_NUMBER: _ClassVar[int]
    FOCAL_LENGTH_X_FIELD_NUMBER: _ClassVar[int]
    FOCAL_LENGTH_Y_FIELD_NUMBER: _ClassVar[int]
    RADIAL_COEFFS_FIELD_NUMBER: _ClassVar[int]
    TANGENTIAL_COEFFS_FIELD_NUMBER: _ClassVar[int]
    THIN_PRISM_COEFFS_FIELD_NUMBER: _ClassVar[int]
    principal_point_x: float
    principal_point_y: float
    focal_length_x: float
    focal_length_y: float
    radial_coeffs: _containers.RepeatedScalarFieldContainer[float]
    tangential_coeffs: _containers.RepeatedScalarFieldContainer[float]
    thin_prism_coeffs: _containers.RepeatedScalarFieldContainer[float]
    def __init__(
        self,
        principal_point_x: _Optional[float] = ...,
        principal_point_y: _Optional[float] = ...,
        focal_length_x: _Optional[float] = ...,
        focal_length_y: _Optional[float] = ...,
        radial_coeffs: _Optional[_Iterable[float]] = ...,
        tangential_coeffs: _Optional[_Iterable[float]] = ...,
        thin_prism_coeffs: _Optional[_Iterable[float]] = ...,
    ) -> None: ...

class OpenCVFisheyeCameraParam(_message.Message):
    __slots__ = (
        "principal_point_x",
        "principal_point_y",
        "focal_length_x",
        "focal_length_y",
        "radial_coeffs",
        "max_angle",
    )
    PRINCIPAL_POINT_X_FIELD_NUMBER: _ClassVar[int]
    PRINCIPAL_POINT_Y_FIELD_NUMBER: _ClassVar[int]
    FOCAL_LENGTH_X_FIELD_NUMBER: _ClassVar[int]
    FOCAL_LENGTH_Y_FIELD_NUMBER: _ClassVar[int]
    RADIAL_COEFFS_FIELD_NUMBER: _ClassVar[int]
    MAX_ANGLE_FIELD_NUMBER: _ClassVar[int]
    principal_point_x: float
    principal_point_y: float
    focal_length_x: float
    focal_length_y: float
    radial_coeffs: _containers.RepeatedScalarFieldContainer[float]
    max_angle: float
    def __init__(
        self,
        principal_point_x: _Optional[float] = ...,
        principal_point_y: _Optional[float] = ...,
        focal_length_x: _Optional[float] = ...,
        focal_length_y: _Optional[float] = ...,
        radial_coeffs: _Optional[_Iterable[float]] = ...,
        max_angle: _Optional[float] = ...,
    ) -> None: ...

class BivariateWindshieldModelParameters(_message.Message):
    __slots__ = (
        "reference_poly",
        "horizontal_poly",
        "vertical_poly",
        "horizontal_poly_inverse",
        "vertical_poly_inverse",
    )
    class ReferencePolynomial(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        FORWARD: _ClassVar[BivariateWindshieldModelParameters.ReferencePolynomial]
        BACKWARD: _ClassVar[BivariateWindshieldModelParameters.ReferencePolynomial]

    FORWARD: BivariateWindshieldModelParameters.ReferencePolynomial
    BACKWARD: BivariateWindshieldModelParameters.ReferencePolynomial
    REFERENCE_POLY_FIELD_NUMBER: _ClassVar[int]
    HORIZONTAL_POLY_FIELD_NUMBER: _ClassVar[int]
    VERTICAL_POLY_FIELD_NUMBER: _ClassVar[int]
    HORIZONTAL_POLY_INVERSE_FIELD_NUMBER: _ClassVar[int]
    VERTICAL_POLY_INVERSE_FIELD_NUMBER: _ClassVar[int]
    reference_poly: BivariateWindshieldModelParameters.ReferencePolynomial
    horizontal_poly: _containers.RepeatedScalarFieldContainer[float]
    vertical_poly: _containers.RepeatedScalarFieldContainer[float]
    horizontal_poly_inverse: _containers.RepeatedScalarFieldContainer[float]
    vertical_poly_inverse: _containers.RepeatedScalarFieldContainer[float]
    def __init__(
        self,
        reference_poly: _Optional[_Union[BivariateWindshieldModelParameters.ReferencePolynomial, str]] = ...,
        horizontal_poly: _Optional[_Iterable[float]] = ...,
        vertical_poly: _Optional[_Iterable[float]] = ...,
        horizontal_poly_inverse: _Optional[_Iterable[float]] = ...,
        vertical_poly_inverse: _Optional[_Iterable[float]] = ...,
    ) -> None: ...

class CameraSpec(_message.Message):
    __slots__ = (
        "temporary_camera_spec",
        "ftheta_param",
        "opencv_pinhole_param",
        "opencv_fisheye_param",
        "logical_id",
        "trajectory_idx",
        "resolution_h",
        "resolution_w",
        "shutter_type",
        "bivariate_windshield_model_param",
    )
    TEMPORARY_CAMERA_SPEC_FIELD_NUMBER: _ClassVar[int]
    FTHETA_PARAM_FIELD_NUMBER: _ClassVar[int]
    OPENCV_PINHOLE_PARAM_FIELD_NUMBER: _ClassVar[int]
    OPENCV_FISHEYE_PARAM_FIELD_NUMBER: _ClassVar[int]
    LOGICAL_ID_FIELD_NUMBER: _ClassVar[int]
    TRAJECTORY_IDX_FIELD_NUMBER: _ClassVar[int]
    RESOLUTION_H_FIELD_NUMBER: _ClassVar[int]
    RESOLUTION_W_FIELD_NUMBER: _ClassVar[int]
    SHUTTER_TYPE_FIELD_NUMBER: _ClassVar[int]
    BIVARIATE_WINDSHIELD_MODEL_PARAM_FIELD_NUMBER: _ClassVar[int]
    temporary_camera_spec: str
    ftheta_param: FthetaCameraParam
    opencv_pinhole_param: OpenCVPinholeCameraParam
    opencv_fisheye_param: OpenCVFisheyeCameraParam
    logical_id: str
    trajectory_idx: int
    resolution_h: int
    resolution_w: int
    shutter_type: ShutterType
    bivariate_windshield_model_param: BivariateWindshieldModelParameters
    def __init__(
        self,
        temporary_camera_spec: _Optional[str] = ...,
        ftheta_param: _Optional[_Union[FthetaCameraParam, _Mapping]] = ...,
        opencv_pinhole_param: _Optional[_Union[OpenCVPinholeCameraParam, _Mapping]] = ...,
        opencv_fisheye_param: _Optional[_Union[OpenCVFisheyeCameraParam, _Mapping]] = ...,
        logical_id: _Optional[str] = ...,
        trajectory_idx: _Optional[int] = ...,
        resolution_h: _Optional[int] = ...,
        resolution_w: _Optional[int] = ...,
        shutter_type: _Optional[_Union[ShutterType, str]] = ...,
        bivariate_windshield_model_param: _Optional[_Union[BivariateWindshieldModelParameters, _Mapping]] = ...,
    ) -> None: ...

class PosePair(_message.Message):
    __slots__ = ("start_pose", "end_pose")
    START_POSE_FIELD_NUMBER: _ClassVar[int]
    END_POSE_FIELD_NUMBER: _ClassVar[int]
    start_pose: _common_pb2.Pose
    end_pose: _common_pb2.Pose
    def __init__(
        self,
        start_pose: _Optional[_Union[_common_pb2.Pose, _Mapping]] = ...,
        end_pose: _Optional[_Union[_common_pb2.Pose, _Mapping]] = ...,
    ) -> None: ...

class DynamicObject(_message.Message):
    __slots__ = ("track_id", "pose_pair")
    TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    POSE_PAIR_FIELD_NUMBER: _ClassVar[int]
    track_id: str
    pose_pair: PosePair
    def __init__(
        self, track_id: _Optional[str] = ..., pose_pair: _Optional[_Union[PosePair, _Mapping]] = ...
    ) -> None: ...

class RGBRenderRequest(_message.Message):
    __slots__ = (
        "scene_id",
        "resolution_h",
        "resolution_w",
        "camera_intrinsics",
        "frame_start_us",
        "frame_end_us",
        "sensor_pose",
        "dynamic_objects",
        "image_format",
        "image_quality",
        "insert_ego_mask",
        "ego_mask_id",
    )
    SCENE_ID_FIELD_NUMBER: _ClassVar[int]
    RESOLUTION_H_FIELD_NUMBER: _ClassVar[int]
    RESOLUTION_W_FIELD_NUMBER: _ClassVar[int]
    CAMERA_INTRINSICS_FIELD_NUMBER: _ClassVar[int]
    FRAME_START_US_FIELD_NUMBER: _ClassVar[int]
    FRAME_END_US_FIELD_NUMBER: _ClassVar[int]
    SENSOR_POSE_FIELD_NUMBER: _ClassVar[int]
    DYNAMIC_OBJECTS_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FORMAT_FIELD_NUMBER: _ClassVar[int]
    IMAGE_QUALITY_FIELD_NUMBER: _ClassVar[int]
    INSERT_EGO_MASK_FIELD_NUMBER: _ClassVar[int]
    EGO_MASK_ID_FIELD_NUMBER: _ClassVar[int]
    scene_id: str
    resolution_h: int
    resolution_w: int
    camera_intrinsics: CameraSpec
    frame_start_us: int
    frame_end_us: int
    sensor_pose: PosePair
    dynamic_objects: _containers.RepeatedCompositeFieldContainer[DynamicObject]
    image_format: ImageFormat
    image_quality: float
    insert_ego_mask: bool
    ego_mask_id: EgoMaskId
    def __init__(
        self,
        scene_id: _Optional[str] = ...,
        resolution_h: _Optional[int] = ...,
        resolution_w: _Optional[int] = ...,
        camera_intrinsics: _Optional[_Union[CameraSpec, _Mapping]] = ...,
        frame_start_us: _Optional[int] = ...,
        frame_end_us: _Optional[int] = ...,
        sensor_pose: _Optional[_Union[PosePair, _Mapping]] = ...,
        dynamic_objects: _Optional[_Iterable[_Union[DynamicObject, _Mapping]]] = ...,
        image_format: _Optional[_Union[ImageFormat, str]] = ...,
        image_quality: _Optional[float] = ...,
        insert_ego_mask: bool = ...,
        ego_mask_id: _Optional[_Union[EgoMaskId, _Mapping]] = ...,
    ) -> None: ...

class AvailableCamerasRequest(_message.Message):
    __slots__ = ("scene_id",)
    SCENE_ID_FIELD_NUMBER: _ClassVar[int]
    scene_id: str
    def __init__(self, scene_id: _Optional[str] = ...) -> None: ...

class AvailableCamerasReturn(_message.Message):
    __slots__ = ("available_cameras",)
    class AvailableCamera(_message.Message):
        __slots__ = ("intrinsics", "rig_to_camera", "logical_id", "trajectory_idx")
        INTRINSICS_FIELD_NUMBER: _ClassVar[int]
        RIG_TO_CAMERA_FIELD_NUMBER: _ClassVar[int]
        LOGICAL_ID_FIELD_NUMBER: _ClassVar[int]
        TRAJECTORY_IDX_FIELD_NUMBER: _ClassVar[int]
        intrinsics: CameraSpec
        rig_to_camera: _common_pb2.Pose
        logical_id: str
        trajectory_idx: int
        def __init__(
            self,
            intrinsics: _Optional[_Union[CameraSpec, _Mapping]] = ...,
            rig_to_camera: _Optional[_Union[_common_pb2.Pose, _Mapping]] = ...,
            logical_id: _Optional[str] = ...,
            trajectory_idx: _Optional[int] = ...,
        ) -> None: ...

    AVAILABLE_CAMERAS_FIELD_NUMBER: _ClassVar[int]
    available_cameras: _containers.RepeatedCompositeFieldContainer[AvailableCamerasReturn.AvailableCamera]
    def __init__(
        self, available_cameras: _Optional[_Iterable[_Union[AvailableCamerasReturn.AvailableCamera, _Mapping]]] = ...
    ) -> None: ...

class AvailableTrajectoriesRequest(_message.Message):
    __slots__ = ("scene_id",)
    SCENE_ID_FIELD_NUMBER: _ClassVar[int]
    scene_id: str
    def __init__(self, scene_id: _Optional[str] = ...) -> None: ...

class AvailableTrajectoriesReturn(_message.Message):
    __slots__ = ("available_trajectories",)
    class AvailableTrajectory(_message.Message):
        __slots__ = ("trajectory_idx", "trajectory")
        TRAJECTORY_IDX_FIELD_NUMBER: _ClassVar[int]
        TRAJECTORY_FIELD_NUMBER: _ClassVar[int]
        trajectory_idx: int
        trajectory: _common_pb2.Trajectory
        def __init__(
            self,
            trajectory_idx: _Optional[int] = ...,
            trajectory: _Optional[_Union[_common_pb2.Trajectory, _Mapping]] = ...,
        ) -> None: ...

    AVAILABLE_TRAJECTORIES_FIELD_NUMBER: _ClassVar[int]
    available_trajectories: _containers.RepeatedCompositeFieldContainer[AvailableTrajectoriesReturn.AvailableTrajectory]
    def __init__(
        self,
        available_trajectories: _Optional[
            _Iterable[_Union[AvailableTrajectoriesReturn.AvailableTrajectory, _Mapping]]
        ] = ...,
    ) -> None: ...

class ExternalAssetObjectsRequest(_message.Message):
    __slots__ = ("scene_id",)
    SCENE_ID_FIELD_NUMBER: _ClassVar[int]
    scene_id: str
    def __init__(self, scene_id: _Optional[str] = ...) -> None: ...

class ExternalAssetObjectsReturn(_message.Message):
    __slots__ = ("track_ids",)
    TRACK_IDS_FIELD_NUMBER: _ClassVar[int]
    track_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, track_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class RGBRenderReturn(_message.Message):
    __slots__ = ("image_bytes",)
    IMAGE_BYTES_FIELD_NUMBER: _ClassVar[int]
    image_bytes: bytes
    def __init__(self, image_bytes: _Optional[bytes] = ...) -> None: ...

class BatchRGBRenderRequest(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[BatchRGBRenderRequestItem]
    def __init__(self, items: _Optional[_Iterable[_Union[BatchRGBRenderRequestItem, _Mapping]]] = ...) -> None: ...

class BatchRGBRenderRequestItem(_message.Message):
    __slots__ = ("camera_name", "request")
    CAMERA_NAME_FIELD_NUMBER: _ClassVar[int]
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    camera_name: str
    request: RGBRenderRequest
    def __init__(
        self, camera_name: _Optional[str] = ..., request: _Optional[_Union[RGBRenderRequest, _Mapping]] = ...
    ) -> None: ...

class BatchRGBRenderReturn(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[BatchRGBRenderReturnItem]
    def __init__(self, items: _Optional[_Iterable[_Union[BatchRGBRenderReturnItem, _Mapping]]] = ...) -> None: ...

class BatchRGBRenderReturnItem(_message.Message):
    __slots__ = ("camera_name", "result", "success", "error_message")
    CAMERA_NAME_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    camera_name: str
    result: RGBRenderReturn
    success: bool
    error_message: str
    def __init__(
        self,
        camera_name: _Optional[str] = ...,
        result: _Optional[_Union[RGBRenderReturn, _Mapping]] = ...,
        success: bool = ...,
        error_message: _Optional[str] = ...,
    ) -> None: ...

class LidarSpec(_message.Message):
    __slots__ = ("lidar_type",)
    LIDAR_TYPE_FIELD_NUMBER: _ClassVar[int]
    lidar_type: LidarDeviceType
    def __init__(self, lidar_type: _Optional[_Union[LidarDeviceType, str]] = ...) -> None: ...

class LidarRenderFilter(_message.Message):
    __slots__ = ("raydrop_threshold", "opacity_threshold", "enable_distance_filter", "distance_filter_threshold")
    RAYDROP_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    OPACITY_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    ENABLE_DISTANCE_FILTER_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_FILTER_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    raydrop_threshold: float
    opacity_threshold: float
    enable_distance_filter: bool
    distance_filter_threshold: float
    def __init__(
        self,
        raydrop_threshold: _Optional[float] = ...,
        opacity_threshold: _Optional[float] = ...,
        enable_distance_filter: bool = ...,
        distance_filter_threshold: _Optional[float] = ...,
    ) -> None: ...

class LidarRenderReturn(_message.Message):
    __slots__ = ("point_xyzs", "point_intensities", "num_points", "point_xyzs_buffer", "point_intensities_buffer")
    POINT_XYZS_FIELD_NUMBER: _ClassVar[int]
    POINT_INTENSITIES_FIELD_NUMBER: _ClassVar[int]
    NUM_POINTS_FIELD_NUMBER: _ClassVar[int]
    POINT_XYZS_BUFFER_FIELD_NUMBER: _ClassVar[int]
    POINT_INTENSITIES_BUFFER_FIELD_NUMBER: _ClassVar[int]
    point_xyzs: _containers.RepeatedScalarFieldContainer[float]
    point_intensities: _containers.RepeatedScalarFieldContainer[float]
    num_points: int
    point_xyzs_buffer: bytes
    point_intensities_buffer: bytes
    def __init__(
        self,
        point_xyzs: _Optional[_Iterable[float]] = ...,
        point_intensities: _Optional[_Iterable[float]] = ...,
        num_points: _Optional[int] = ...,
        point_xyzs_buffer: _Optional[bytes] = ...,
        point_intensities_buffer: _Optional[bytes] = ...,
    ) -> None: ...

class LidarRenderRequest(_message.Message):
    __slots__ = (
        "scene_id",
        "lidar_config",
        "frame_start_us",
        "frame_end_us",
        "sensor_pose",
        "dynamic_objects",
        "render_filter",
    )
    SCENE_ID_FIELD_NUMBER: _ClassVar[int]
    LIDAR_CONFIG_FIELD_NUMBER: _ClassVar[int]
    FRAME_START_US_FIELD_NUMBER: _ClassVar[int]
    FRAME_END_US_FIELD_NUMBER: _ClassVar[int]
    SENSOR_POSE_FIELD_NUMBER: _ClassVar[int]
    DYNAMIC_OBJECTS_FIELD_NUMBER: _ClassVar[int]
    RENDER_FILTER_FIELD_NUMBER: _ClassVar[int]
    scene_id: str
    lidar_config: LidarSpec
    frame_start_us: int
    frame_end_us: int
    sensor_pose: PosePair
    dynamic_objects: _containers.RepeatedCompositeFieldContainer[DynamicObject]
    render_filter: LidarRenderFilter
    def __init__(
        self,
        scene_id: _Optional[str] = ...,
        lidar_config: _Optional[_Union[LidarSpec, _Mapping]] = ...,
        frame_start_us: _Optional[int] = ...,
        frame_end_us: _Optional[int] = ...,
        sensor_pose: _Optional[_Union[PosePair, _Mapping]] = ...,
        dynamic_objects: _Optional[_Iterable[_Union[DynamicObject, _Mapping]]] = ...,
        render_filter: _Optional[_Union[LidarRenderFilter, _Mapping]] = ...,
    ) -> None: ...

class DynamicObjectTrack(_message.Message):
    __slots__ = ("id", "semantic_class", "trajectory", "object_size", "asset_id")
    ID_FIELD_NUMBER: _ClassVar[int]
    SEMANTIC_CLASS_FIELD_NUMBER: _ClassVar[int]
    TRAJECTORY_FIELD_NUMBER: _ClassVar[int]
    OBJECT_SIZE_FIELD_NUMBER: _ClassVar[int]
    ASSET_ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    semantic_class: str
    trajectory: _common_pb2.Trajectory
    object_size: _common_pb2.AABB
    asset_id: str
    def __init__(
        self,
        id: _Optional[str] = ...,
        semantic_class: _Optional[str] = ...,
        trajectory: _Optional[_Union[_common_pb2.Trajectory, _Mapping]] = ...,
        object_size: _Optional[_Union[_common_pb2.AABB, _Mapping]] = ...,
        asset_id: _Optional[str] = ...,
    ) -> None: ...

class AvailableDynamicObjectsRequest(_message.Message):
    __slots__ = ("scene_id",)
    SCENE_ID_FIELD_NUMBER: _ClassVar[int]
    scene_id: str
    def __init__(self, scene_id: _Optional[str] = ...) -> None: ...

class AvailableDynamicObjectsReturn(_message.Message):
    __slots__ = ("dynamic_objects",)
    DYNAMIC_OBJECTS_FIELD_NUMBER: _ClassVar[int]
    dynamic_objects: _containers.RepeatedCompositeFieldContainer[DynamicObjectTrack]
    def __init__(self, dynamic_objects: _Optional[_Iterable[_Union[DynamicObjectTrack, _Mapping]]] = ...) -> None: ...

class ReplaceAssetAction(_message.Message):
    __slots__ = ("original_id", "replacement_id", "object_size")
    ORIGINAL_ID_FIELD_NUMBER: _ClassVar[int]
    REPLACEMENT_ID_FIELD_NUMBER: _ClassVar[int]
    OBJECT_SIZE_FIELD_NUMBER: _ClassVar[int]
    original_id: str
    replacement_id: str
    object_size: _common_pb2.AABB
    def __init__(
        self,
        original_id: _Optional[str] = ...,
        replacement_id: _Optional[str] = ...,
        object_size: _Optional[_Union[_common_pb2.AABB, _Mapping]] = ...,
    ) -> None: ...

class EditAssetsRequest(_message.Message):
    __slots__ = ("scene_id", "replace", "insert")
    SCENE_ID_FIELD_NUMBER: _ClassVar[int]
    REPLACE_FIELD_NUMBER: _ClassVar[int]
    INSERT_FIELD_NUMBER: _ClassVar[int]
    scene_id: str
    replace: _containers.RepeatedCompositeFieldContainer[ReplaceAssetAction]
    insert: _containers.RepeatedCompositeFieldContainer[DynamicObjectTrack]
    def __init__(
        self,
        scene_id: _Optional[str] = ...,
        replace: _Optional[_Iterable[_Union[ReplaceAssetAction, _Mapping]]] = ...,
        insert: _Optional[_Iterable[_Union[DynamicObjectTrack, _Mapping]]] = ...,
    ) -> None: ...

class EditAssetsResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

class RestoreModelParametersRequest(_message.Message):
    __slots__ = ("scene_id",)
    SCENE_ID_FIELD_NUMBER: _ClassVar[int]
    scene_id: str
    def __init__(self, scene_id: _Optional[str] = ...) -> None: ...

class ServerConfig(_message.Message):
    __slots__ = ("server_config",)
    class ServerConfigEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

    SERVER_CONFIG_FIELD_NUMBER: _ClassVar[int]
    server_config: _containers.ScalarMap[str, str]
    def __init__(self, server_config: _Optional[_Mapping[str, str]] = ...) -> None: ...
