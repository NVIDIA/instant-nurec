// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#pragma once

#include <nrend/allocatorParameters.h>
#include <nrend/errorCodes.h>
#include <nrend/loggerParameters.h>
#include <nrend/renderingParameters.h>

#ifdef _MSC_VER
#ifdef nrend_EXPORTS
#define NREND_ABI __declspec(dllexport)
#else
#define NREND_ABI __declspec(dllimport)
#endif
#else
#ifdef nrend_EXPORTS
#define NREND_ABI __attribute__((visibility("default")))
#else
#define NREND_ABI
#endif
#endif
namespace nrend {

/**
 * @brief Neural Reconstruction Rendering Interface
 *
 * @details
 * INRenderer defines a differentiable rendering interface for rendering models optimized with the Neural Rendering Engine (NRE) framework
 * https://gitlab-master.nvidia.com/nrs/nre
 */
struct NREND_ABI INRenderer {

    /**
     * @brief A simple vector type
     */
    template <typename T, uint32_t N, size_t ALIGNMENT = sizeof(T)>
    struct alignas(ALIGNMENT) TVec {
        T elems[N];
    };
    using Vec2  = TVec<float, 2>;
    using Vec3  = TVec<float, 3>;
    using Vec4  = TVec<float, 4>;
    using IVec2 = TVec<int32_t, 2>;
    using UVec4 = TVec<uint32_t, 4>;

    /**
     * @brief A simple matrix type castable into tcnn::tmat
     */
    template <typename T, uint32_t N, uint32_t M>
    struct TMat {
        union {
            TVec<T, M> m[N];
            T d[M * N];
        };

        static inline TMat<T, N, M> identity() {
            TMat<T, N, M> matrix;
            for (uint32_t i = 0; i < N; ++i) {
                for (uint32_t j = 0; j < M; ++j) {
                    matrix.m[i].elems[j] = (i == j ? (T)1 : (T)0);
                }
            }
            return matrix;
        };
    };
    using Mat4x3 = TMat<float, 4, 3>;

    /**
     * @brief A 3D Bounding Box castable into tcnn::BoundingBox
     */
    struct BoundingBox {
        Vec3 min;
        Vec3 max;
    };

    /**
     * @brief Timestamp
     */
    using TTimestamp = int64_t;

    /**
     * @brief Pose : 3d position | 3d rotation as a xyzw quaternion
     */
    using TTrackInstancePose = TVec<float, 7>;
    using TSensorPose        = TVec<float, 7>;

    /**
     * @brief SensorState
     */
    struct TSensorState {
        TTimestamp startTimestamp;
        TSensorPose startPose;
        TTimestamp endTimestamp;
        TSensorPose endPose;

        static inline TSensorPose poseIdentity() {
            return {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f};
        };

        static TSensorPose poseInverse(const TSensorPose& pose);
    };

    /**
     * @brief Sensor type
     */
    enum SensorType {
        Camera,
        Lidar
    };

    /**
     * @brief Orthographic projection parameters
     * @note : in viewport resolution
     */
    struct OrthographicProjectionParameters {
    };

    /**
     * @brief Perspective projection parameters
     * @note : in viewport resolution
     */
    struct PerspectiveProjectionParameters {
        Vec2 principalPoint;
        Vec2 focalLength;
    };

    /**
     * @brief OpenCV pinhole with radial distortion projection parameters
     * @note : in nominal resolution
     */
    struct OpenCVPinholeProjectionParameters {
        Vec2 nominalResolution;
        Vec2 principalPoint;
        Vec2 focalLength;
        TVec<float, 6> radialCoeffs;
        Vec2 tangentialCoeffs;
        Vec4 thinPrismCoeffs;
    };

    /**
     * @brief OpenCV fisheye projection parameters
     * @note : in nominal resolution
     */
    struct OpenCVFisheyeProjectionParameters {
        Vec2 nominalResolution;
        Vec2 principalPoint;
        Vec2 focalLength;
        Vec4 radialCoeffs;
        float maxAngle;
    };

    /**
     * @brief FTheta projection parameters
     * @note : in nominal resolution
     */
    struct FThetaProjectionParameters {
        Vec2 nominalResolution;
        Vec2 principalPoint;
        enum PolynomialType {
            PIXELDIST_TO_ANGLE,
            ANGLE_TO_PIXELDIST,
            PIXELDIST_TO_ANGLE_RF
        } referencePoly;
        static constexpr size_t PolynomialDegree = 6;
        TVec<float, PolynomialDegree> pixeldistToAnglePoly; ///< backward polynomial
        TVec<float, PolynomialDegree> angleToPixeldistPoly; ///< forward polynomial
        float maxAngle;
        Vec3 linear_cde;
    };

    /**
     * @brief NRE HesaiP128 Lidar projection parameters
     * @note : fixed resolution
     */
    struct RowOffsetStructuredSpinningLidarProjectionParameters {
        enum SpinningDirection {
            CLOCK_WISE,
            COUNTER_CLOCK_WISE
        } spin;
        int32_t nRows;
        int32_t nColumns;
        Vec2 fovStart; ///< Start of (azimuth, elevation) field-of-view range (radians)
        Vec2 fovSpan;  ///< Span of (azimuth, elevation) field-of-view range (radians)
        int32_t azimuthNBins;
        int32_t elevationNBins;
        int32_t maxPtsPerTile;
        const IVec2* tilesPackInfo;      ///< Device buffer
        const IVec2* tilesToElementsMap; ///< Device buffer
        int32_t elevationCDFResolution;
        int32_t azimuthCDFResolution;
        const int* elevationCDFTable;    ///< Device data of the elevation CDF table [elevationCDFResolution + 1]
        const int* denseRayMaskCDFTable; ///< Device buffer of size (azimuthCDFResolution + 1), (elevationCDFResolution + 1)
        int angleToColumnMapResolutionFactor;
        Vec2 mapResolution;
        const int* angleToColumnMap; ///< Device buffer
    };

    /**
     * @brief Generalized projection parameters
     * @note : NDC resolution
     */
    struct GeneralizedProjectionParameters {
        const Vec3* projectionMap;
        const Vec2* octahedralUnprojectMap;
        // TODO : add nominal width/heigh, optical center, fov
    };

    /**
     * @brief Bivariate windshield distortion parameters
     */
    struct BivariateWindshieldDistortionParameters {
        static constexpr int32_t N_MAX_POLY_ORDER = 5; // max poly order should be increased as requires
        static constexpr int32_t N_MAX_POLY_SIZE  = (N_MAX_POLY_ORDER + 2) * (N_MAX_POLY_ORDER + 1) / 2;
        const float* horizontalPoly;
        const float* verticalPoly;
        int32_t horizontalPolyOrder;
        int32_t verticalPolyOrder;
    };

    /**
     * @brief Sensor projection parameters
     */
    struct SensorProjectionModel {

        enum ShutterType {
            RollingTopToBottomShutter,
            RollingLeftToRightShutter,
            RollingBottomToTopShutter,
            RollingRightToLeftShutter,
            GlobalShutter,
            Undefined,
        } shutterType = GlobalShutter;

        enum ModelType {
            OrthographicModel,
            PerspectiveModel,
            OpenCVPinholeModel,
            OpenCVFisheyeModel,
            FThetaModel,
            RowOffsetStructuredSpinningLidarModel,
            GeneralizedModel,
            EmptyModel,
            Unsupported
        } modelType = EmptyModel;

        union {
            OrthographicProjectionParameters orthographicParams;
            PerspectiveProjectionParameters perspectiveParams;
            OpenCVPinholeProjectionParameters ocvPinholeParams;
            OpenCVFisheyeProjectionParameters ocvFisheyeParams;
            FThetaProjectionParameters fthetaParams;
            RowOffsetStructuredSpinningLidarProjectionParameters nreHesaiP128LidarParams;
            GeneralizedProjectionParameters generalizedParams;
        };

        enum ExternalDistortionType {
            EmptyExternalDistortionModel,
            BivariateWindshieldDistortion,
            UnsupportedExternalDistortion
        } externalDistortionType = EmptyExternalDistortionModel;

        union {
            BivariateWindshieldDistortionParameters bivariateWindshieldDistortionParameters;
        };
    };
    using TSensorModel = SensorProjectionModel;

    /// @brief opaque handle to an instance of a renderer
    using RendererHandle = uint64_t;

    /// @brief invalid renderer handle
    static constexpr RendererHandle InvalidRendererHandle = 0;

    /// @brief opaque handle to a device queue (castable to cudaStream_t)
    using DeviceQueueHandle = uint64_t;

    /**
     * Create a renderer given a model and a renderer specifications
     *
     * @param modelSpecificationData        Input host msgpack data span containing the model specification
     * @param rendererSpecificationData     Input host msgpack data span containing the renderer specification.
     *                                      If empty, use the default renderer for the given model
     * @param renderingParameters           Input rendering parameters
     * @param loggerParameters              Input logger parameters
     * @param handle                        Output handle to the created engine
     */
    static ErrorCode create(const MsgPackData& modelSpecificationData,
                            const MsgPackData& renderSpecificationData,
                            const RenderingParameters& rendererParameters,
                            const LoggerParameters& loggerSpecifications,
                            RendererHandle& handle);

    /**
     * Create a renderer given a model and a renderer specifications
     *
     * @param handle                        Input opaque handle of the renderer to get the model version
     * @param versionMajor                  Output major version of the model
     * @param versionMinor                  Output minor version of the model
     * @param versionPatch                  Output patch version of the model
     * @param modelName                     Output name of the model buffer (valid until the renderer is destroyed)
     */
    static ErrorCode getModelVersion(RendererHandle handle,
                                     int& versionMajor,
                                     int& versionMinor,
                                     int& versionPatch,
                                     const char*& modelName);

    /**
     * @brief Get the rendering features layout. The features layout defines the dimensions of the features buffers to be provided to the rendering functions.
     *
     * @param handle          Input opaque handle of the renderer
     * @param sensorType      Input sensor type
     * @param featuresLayout  Output features layout specifying the dimensions of the features buffers to be provided to the rendering functions
     */
    static ErrorCode renderingFeaturesLayout(RendererHandle handle,
                                             SensorType sensorType,
                                             RenderingFeaturesLayout& featuresLayout);

    /**
     * @brief Get the layout of the scene data buffer to be provided to the rendering functions.
     *
     * @param handle           Input opaque handle of the renderer
     * @param sceneDataSize    Output number of elements of the scene data buffer to be provided to the rendering functions
     * @param sceneDataLayout  Output layout of the scene data buffer to be provided to the rendering functions
     */
    static ErrorCode renderingSceneDataLayout(RendererHandle handle,
                                              uint32_t& sceneDataSize,
                                              RenderingSceneDataLayout& sceneDataLayout);

    /**
     * Destroy a renderer.
     * @param handle    Input opaque handle of the renderer to destroy
     */
    static void destroy(RendererHandle handle);

    /// @brief opaque handle to a rendering context used for backward propagation
    using RenderingContextHandle = uint64_t;

    /// @brief invalid renderering context handle
    static constexpr RenderingContextHandle InvalidRenderingContextHandle = 0;

    /**
     * @brief Rendering parameters
     */
    struct RenderParameters {
        uint32_t id;                     ///< id of the frame to be rendered
        Vec2 frameResolution;            ///< resolution (width, height) of the frame UV space
        Vec2 frameTileOffset;            ///< offset of the tile in the frame UV space
        IVec2 frameTileResolution;       ///< resolution (width, height) of the tile to be rendered
        float hitTransmittance;          ///< transmittance threshold below which we assume a surface has been hit
        BoundingBox objectAABB;          ///< 3D bounding box defining the extent of the object in the scene
        Mat4x3 worldToObjectTransform;   ///< to object transform
        Mat4x3 objectToWorldTransform;   ///< to world transform
        TSensorModel sensorModel;        ///< parameters of the sensor
        TSensorState sensorState;        ///< state of the sensor
        Mat4x3 colorCorrectionMatrix;    ///< color matrix to be applied
        UVec4 objectInstanceIds;         ///< up-to 4 object instance ids to be ignored by the depth composition
        int32_t numActiveTrackInstances; ///< number of active track instances

        // NOTE: This value is not used by NRE, but is kept for compatibility with the Kit.
        static constexpr float defaultHitTransmittance = 0.475f; ///< default value for the hit transmittance threshold
    };

    /**
     * Render (forward)
     *
     * @param handle                                Input opaque handle of the renderer
     * @param params                                Input render parameters
     * @param worldRayOriginDevicePtr               Input buffer containing the rays origin in world space
     * @param worldRayDirectionDevicePtr            Input buffer containing the rays direction in world space (norm is the ray spread)
     * @param worldRayTimestampDevicePtr            Input buffer containing the ray timestamps
     * @param sensorsIdsDevicePtr                   Input buffer containing the sensor ids [sensor id, frame id]
     * @param activeTrackInstancesIdsDevicePtr      Input buffer containing the ids of the active track instances
     * @param activeTrackInstancesPoseDevicePtr     Input buffer containing the start pose of the active track instances
     * @param activeTrackInstancesEndPoseDevicePtr  Input buffer containing the end pose of the active track instances
     * @param instanceIdDevicePtr                   Input/Output buffer containing the rendered instance ids
     * @param worldHitDistanceDevicePtr             Input/Output buffer containing the rendered depth
     * @param worldHitNormalDevicePtr               Input/Output buffer containing the rendered normal
     * @param radianceDensityDevicePtr              Input/Output buffer containing the rendered radiance and opacity [H x W x (baseFeatureDim + 1)]
     * @param extendedFeaturesDevicePtr             Input/Output buffer containing the rendered extended features [H x W x extendedFeatureDim]
     * @param deviceIndex                           Input index of the device
     * @param deviceQueue                           Input device queue to be used for copying the buffer if required
     * @param context                               Output context of the rendering call
     */
    static ErrorCode render(RendererHandle handle,
                            const RenderParameters& params,
                            const Vec3* worldRayOriginCudaPtr,
                            const Vec3* worldRayDirectionCudaPtr,
                            const TTimestamp* worldRayTimestampCudaPtr,
                            const IVec2* sensorsIdsCudaPtr,
                            const IVec2* activeTrackInstancesIdsCudaPtr,
                            const TTrackInstancePose* activeTrackInstancesPoseCudaPtr,
                            const TTrackInstancePose* activeTrackInstancesEndPoseCudaPtr,
                            uint32_t* instanceIdCudaPtr,
                            float* worldHitDistanceCudaPtr,
                            Vec3* worldHitNormalCudaPtr,
                            Vec4* radianceDensityCudaPtr,
                            void* extendedFeaturesCudaPtr,
                            void* sceneDataCudaPtr,
                            int deviceIndex,
                            DeviceQueueHandle deviceQueue,
                            RenderingContextHandle* context);

    /**
     * Render backward
     *
     * @param handle                                Input opaque handle of the renderer
     * @param params                                Input render parameters
     * @param worldRayOriginCudaPtr                 Input buffer containing the rays origin in world space
     * @param worldRayDirectionCudaPtr              Input buffer containing the rays direction in world space (norm is the ray spread)
     * @param worldRayTimestampCudaPtr              Input buffer containing the ray timestamps
     * @param sensorsIdsCudaPtr                     Input buffer containing the sensor ids [sensor id, frame id]
     * @param activeTrackInstancesIdsCudaPtr        Input buffer containing the ids of the active track instances
     * @param activeTrackInstancesPoseCudaPtr       Input buffer containing the start pose of the active track instances
     * @param activeTrackInstancesEndPoseCudaPtr    Input buffer containing the end pose of the active track instances
     * @param instanceIdCudaPtr                     Input buffer containing the rendered instance ids
     * @param worldHitDistanceCudaPtr               Input buffer containing the rendered depth
     * @param worldHitDistanceGradientCudaPtr       Input buffer containing the rendered depth gradient
     * @param worldHitNormalCudaPtr                 Input buffer containing the rendered normal
     * @param worldHitNormalGradientCudaPtr         Input buffer containing the rendered normal gradient
     * @param radianceDensityCudaPtr                Input buffer containing the rendered radiance and opacity [H x W x (baseFeatureDim + 1)]
     * @param radianceDensityGradientCudaPtr        Input buffer containing the rendered radiance and opacity gradients [H x W x (baseFeatureDim + 1)]
     * @param extendedFeaturesCudaPtr               Input buffer containing the rendered extended features [H x W x extendedFeatureDim]
     * @param extendedFeaturesGradientCudaPtr       Input buffer containing the rendered extended features gradients [H x W x extendedFeatureDim]
     * @param wordlRayOriginGradientCudaPtr         Input/Output buffer containing he rays origin in world space gradients
     * @param worldRayDirectionGradientCudaPtr      Input/Output buffer containing the rays direction in world space gradients
     * @param deviceIndex                           Input index of the device
     * @param deviceQueue                           Input device queue to be used for copying the buffer if required
     * @param contex                                Input forward rendering call context
     */
    static ErrorCode renderBackward(RendererHandle handle,
                                    const RenderParameters& params,
                                    const Vec3* worldRayOriginCudaPtr,
                                    const Vec3* worldRayDirectionCudaPtr,
                                    const TTimestamp* worldRayTimestampCudaPtr,
                                    const IVec2* sensorsIdsCudaPtr,
                                    const IVec2* activeTrackInstancesIdsCudaPtr,
                                    const TTrackInstancePose* activeTrackInstancesPoseCudaPtr,
                                    const TTrackInstancePose* activeTrackInstancesEndPoseCudaPtr,
                                    uint32_t* instanceIdCudaPtr,
                                    float* worldHitDistanceCudaPtr,
                                    const float* worldHitDistanceGradientCudaPtr,
                                    const Vec3* worldHitNormalCudaPtr,
                                    const Vec3* worldHitNormalGradientCudaPtr,
                                    const Vec4* radianceDensityCudaPtr,
                                    const Vec4* radianceDensityGradientCudaPtr,
                                    const void* extendedFeaturesCudaPtr,
                                    const void* extendedFeaturesGradientCudaPtr,
                                    Vec3* wordlRayOriginGradientCudaPtr,
                                    Vec3* worldRayDirectionGradientCudaPtr,
                                    int deviceIndex,
                                    DeviceQueueHandle deviceQueue,
                                    RenderingContextHandle context);

    /**
     * @brief Get the scene layout. The scene layout defines the dimensions of the scene buffers to be provided to the prepare scene functions.
     *
     * @param handle                        Input opaque handle of the renderer
     * @param sensorType                    Input sensor type
     * @param sceneSize                     Output number of elements in the scene
     * @param sceneDensitySize              Output dimension of the scene density
     * @param featureSize                   Output dimension of the feature
     * @param extendedFeaturesSize          Output dimension of the extended features
     * @param sensorExtendedFeaturesSize    Output dimension of the sensor extended features
     * @param halfPrecision                 Output if the scene buffer are in half precision
     */
    static ErrorCode sceneLayout(RendererHandle handle,
                                 SensorType sensorType,
                                 uint32_t& sceneSize,
                                 uint32_t& sceneDensitySize,
                                 uint32_t& featureSize,
                                 uint32_t& extendedFeaturesSize,
                                 uint32_t& sensorExtendedFeaturesSize,
                                 bool& halfPrecision);

    /**
     * Prepare scene (forward)
     *
     * @param handle                                Input opaque handle to the renderer
     * @param params                                Input render parameters
     * @param activeTrackInstancesIdsCudaPtr        Input buffer containing the ids of the active track instances
     * @param activeTrackInstancesPoseCudaPtr       Input buffer containing the start pose of the active track instances
     * @param activeTrackInstancesEndPoseCudaPtr    Input buffer containing the end pose of the active track instances
     * @param sceneDensityCudaPtr                   Output buffer containing the scene density (size must comply with scene layout)
     * @param sceneFeaturesCudaPtr                  Output buffer containing the scene features (size must comply with scene layout)
     * @param sceneExtendedFeaturesCudaPtr          Output buffer containing the scene extended features (size must comply with scene layout)
     * @param sceneSensorExtendedFeaturesCudaPtr    Output buffer containing the scene sensor extended features (size must comply with scene layout)
     * @param sceneDataCudaPtr                      Output buffer containing the scene data (size must comply with scene layout)
     * @param sceneSize                             Output number of valid elements in the scene buffers
     * @param deviceIndex                           Input index of the device
     * @param deviceQueue                           Input device queue to be used for copying the buffer if required
     * @param forwardContext                        Output context of the prepare scene call
     */
    static ErrorCode prepareScene(RendererHandle handle,
                                  const RenderParameters& params,
                                  const IVec2* activeTrackInstancesIdsCudaPtr,
                                  const TTrackInstancePose* activeTrackInstancesPoseCudaPtr,
                                  const TTrackInstancePose* activeTrackInstancesEndPoseCudaPtr,
                                  void* sceneDensityCudaPtr,
                                  void* sceneFeaturesCudaPtr,
                                  void* sceneExtendedFeaturesCudaPtr,
                                  void* sceneSensorExtendedFeaturesCudaPtr,
                                  void* sceneDataCudaPtr,
                                  uint32_t& sceneSize,
                                  int deviceIndex,
                                  DeviceQueueHandle deviceQueue,
                                  RenderingContextHandle* forwardContext);

    /**
     * Prepare scene (backward)
     *
     * @param handle                                        Input opaque handle to the renderer
     * @param params                                        Input render parameters
     * @param activeTrackInstancesIdsCudaPtr                Input buffer containing the ids of the active track instances
     * @param activeTrackInstancesPoseCudaPtr               Input buffer containing the start pose of the active track instances
     * @param activeTrackInstancesEndPoseCudaPtr            Input buffer containing the end pose of the active track instances
     * @param sceneFeaturesCudaPtr                          Input buffer containing the scene features
     * @param sceneExtendedFeaturesCudaPtr                  Input buffer containing the scene extended features
     * @param sceneSensorExtendedFeaturesCudaPtr            Input buffer containing the scene sensor extended features
     * @param sceneDensityGradientCudaPtr                   Input buffer containing the scene density gradient
     * @param sceneFeaturesGradientCudaPtr                  Input buffer containing the scene features gradient
     * @param sceneExtendedFeaturesGradientCudaPtr          Input buffer containing the scene extended features gradient
     * @param sceneSensorExtendedFeaturesGradientCudaPtr    Input buffer containing the scene sensor extended features gradient
     * @param deviceIndex                                   Input index of the device
     * @param deviceQueue                                   Input device queue to be used for copying the buffer if required
     * @param forwardContext                                Input context of the prepare scene call
     */
    static ErrorCode prepareSceneBackward(RendererHandle handle,
                                          const RenderParameters& params,
                                          const IVec2* activeTrackInstancesIdsCudaPtr,
                                          const TTrackInstancePose* activeTrackInstancesPoseCudaPtr,
                                          const TTrackInstancePose* activeTrackInstancesEndPoseCudaPtr,
                                          const void* sceneFeaturesCudaPtr,
                                          const void* sceneExtendedFeaturesCudaPtr,
                                          const void* sceneSensorExtendedFeaturesCudaPtr,
                                          const void* sceneDensityGradientCudaPtr,
                                          const void* sceneFeaturesGradientCudaPtr,
                                          const void* sceneExtendedFeaturesGradientCudaPtr,
                                          const void* sceneSensorExtendedFeaturesGradientCudaPtr,
                                          int deviceIndex,
                                          DeviceQueueHandle deviceQueue,
                                          RenderingContextHandle forwardContext);

    /**
     * Destroy a render context
     *
     * @param handle        Input opaque handle to the render context
     */
    static void destroyRenderingContext(RenderingContextHandle context);

    /**
     * Update the model parameters buffers on a specific device
     *
     * @param handle        Input opaque handle to the renderer
     * @param namedParametersDefinitions Input dictionary of parameters name, device buffer ptr to be updated
     * @param gradient      Input : if true update only parameter gradient, else parameters values
     * @param copy          Input : if true allocate a buffer and copy the content of the passed one
     * @param deviceIndex   Input index of the device
     * @param deviceQueue   Input device queue to be used for copying the buffer if required
     */
    static ErrorCode updateModelParameters(RendererHandle handle,
                                           const NamedParameterDefinitionsSpan& namedParametersDefinitions,
                                           bool gradients,
                                           bool copy,
                                           int deviceIndex,
                                           DeviceQueueHandle deviceQueue);

    /**
     * Detach all model parameters buffers currently not owned by nrend on a specific device
     *
     * @param handle        Input opaque handle to the renderer
     * @param gradient      Input : if true detach only parameter gradient, else parameters values
     * @param copy          Input : if true allocate a buffer and copy the content of the attached buffer
     * @param deviceIndex   Input index of the device
     * @param deviceQueue   Input device queue to be used for copying the buffer if required
     */
    static ErrorCode detachModelParameters(RendererHandle handle,
                                           bool gradients,
                                           bool copy,
                                           int deviceIndex,
                                           DeviceQueueHandle deviceQueue);

    /**
     * Dependency injection for the allocator that defines allocation and free methods to use in nrend.
     *
     * @param directory     Input allocator to be set in nrend's device allocation structure.
     */
    static ErrorCode setDeviceAllocator(const DeviceMemoryAllocator& allocator);

    /**
     * Set the folder path where the runtime compilation should be cached
     *
     * @param directory     Input path of the directory to be set as rtc cache
     */
    static ErrorCode setRTCCacheDirectory(const char* directory);

    /**
     * Set the folder path where the runtime compilation headers file are located.
     * NB : by default nrend is using internal resources and this folder is not used.
     *
     * @param directory     Input path of the directory to be set as rtc cache
     * @param append        Input : if true append the directory to the existing ones, else replace the existing ones
     * @param extra         Input : if true add the directory to the extra includes which may be obfuscated
     */
    static ErrorCode setRTCIncludeDirectory(const char* directory,
                                            bool append,
                                            bool extra);

    /**
     * Compute the device memory allocated by all renderers.
     *
     * @param usage         Output number of bytes currently allocated by all renderers on all devices
     */
    static ErrorCode devicesMemoryUsage(size_t& usage);
};

} // namespace nrend
