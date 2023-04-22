// Globals
const float PI = 3.1415926535897932384626433832795;
const float INVERSE_PI = 0.3183099; // 1/PI 
float aspectRatio; // defined in main
const float degreesToRads = (PI / 180.0);
const float sunPhiDegrees = degreesToRads * 10.0;   // angle off horizon
const float sunThetaDegrees = degreesToRads * 0.0;  // angle around UP direction
// direction of a beam of light from the sun
const vec3 sun = normalize(vec3(-cos(sunPhiDegrees) * cos(sunThetaDegrees), -sin(sunPhiDegrees) * cos(sunThetaDegrees), -sin(sunThetaDegrees))); 
const float cloudHeight = 15.0;
const mat2 fbmRotateMat = mat2(  0.80,  0.60,
                                -0.60,  0.80 ); // rotation numbers come from 3-4-5 pythagorean triple

const mat2 inverseFbmRotateMat = mat2(0.80, -0.60,
                                      0.60, 0.80);

// Color palette
const vec3 landColor = vec3(0.27, 0.18, 0.15);
const vec3 grassColor = vec3(0.33, 0.50, 0.27);
const vec3 skyColor = vec3(0.35, 0.62, 1.0);
const vec3 cloudColor = vec3(1.0, 1.0, 1.0);
const vec3 hazeColor = vec3(1,1,1);

// Refinement factors
const int terrain_resolution_factor = 9; // number of layers of increasingly higher frequency noise
const float domainNoiseScaleFactor = 5.0;      // controls domain scale of noise pattern
const float rangeNoiseScaleFactor = 3.0;       // controls range scale of noise pattern
const vec3 atmosphericDecayFactor = .0020 * vec3( -1.5, -2.2, -5.0);

// Render-target enums
const int mountain_targetID = 1;
const int cloud_targetID = 2;

//====== Helper / Generic functions ======= //

// Takes in two input numbers and spits out a (very crude) pseudo-
// random number.
float pseudoRandom(in vec2 seed) {
    vec2 intermediate = 50.0*fract(seed*INVERSE_PI);
    return fract((intermediate.x + intermediate.y)*(intermediate.x * intermediate.y));
}

// Even cruder pseudo-random number generator; this time taking in a 1D input
// and an output range.
float pseudoRandomRange(float seed, float minRange, float maxRange) {
    return ((maxRange - minRange) * fract(sin(seed) * 1000000.0)) + minRange;
} 

// Only valid for values of 'pos' between 0 and 1
vec2 smoothstepDeriv(vec2 pos) {
    return 6.0*pos*(1.0-pos);
}


// Continuous, smoothly varying noise function
float noise(in vec2 position) { 
    position /= domainNoiseScaleFactor;  

    vec2 ij = floor(position);
    vec2 posFraction = fract(position);
    vec2 smoothedPositionFract = smoothstep(0.0,1.0,posFraction);

    // Coefficients for noise function
    float a = pseudoRandom(ij + vec2(0,0));
    float b = pseudoRandom(ij + vec2(1,0));
    float c = pseudoRandom(ij + vec2(0,1));
    float d = pseudoRandom(ij + vec2(1,1));    
    
    float noise = a 
        + (b-a)*smoothedPositionFract.x
        + (c-a)*smoothedPositionFract.y 
        + (a-b-c+d)*smoothedPositionFract.x*smoothedPositionFract.y; 
        
    return rangeNoiseScaleFactor * noise;
}

vec2 noiseDeriv(in vec2 position) {
    position /= domainNoiseScaleFactor;

    vec2 ij = floor(position);
    vec2 posFraction = fract(position);    
    vec2 smoothedPositionFract = smoothstep(0.0,1.0,posFraction);
    vec2 dSmoothedPositionFract = smoothstepDeriv(posFraction);

    // Coefficients for noise function
    float a = pseudoRandom(ij + vec2(0,0));
    float b = pseudoRandom(ij + vec2(1,0));
    float c = pseudoRandom(ij + vec2(0,1));
    float d = pseudoRandom(ij + vec2(1,1));  
    
    return rangeNoiseScaleFactor * dSmoothedPositionFract * vec2(
        (b-a) + (a-b-c+d) * smoothedPositionFract.y,
        (c-a) + (a-b-c+d) * smoothedPositionFract.x
    );

}

vec3 rayPosition(in vec3 rayOrigin, in vec3 rayDirection, in float t) {
    return (rayOrigin + (rayDirection * t));
}

// Fractal Brownian Motion - a type of noise made by combining
// different frequency layers of base noise.
float fbm(in vec2 seed, int iterations) {
    float value = noise(seed);
    float domainScale = 1.0;
    float rangeScale = 1.0;
            
    for (int i = 1; i < iterations; i++) {
        domainScale *= 2.0;
        rangeScale /= 2.0;

        // (Instead of matrix * matrix to get a new rotation matrix, computationally cheaper to do mat*vec mult)
        seed = fbmRotateMat * seed;
        
        float value_i = rangeScale * noise(domainScale * seed);
        value += value_i;
    }

    return value;
}

vec2 fbm_deriv(in vec2 seed, int iterations) {
    vec2 derivs = noiseDeriv(seed);
    float domainScale = 1.0;
    float rangeScale = 1.0;

    mat2 inverseRotation = mat2(1, 0, 0, 1);
            
    for (int i = 1; i < iterations; i++) {
        domainScale *= 2.0;
        rangeScale /= 2.0;

        // (Instead of matrix * matrix to get a new rotation matrix, computationally cheaper to do mat*vec mult)
        seed = fbmRotateMat * seed;
        inverseRotation *= inverseFbmRotateMat;
        
        vec2 derivs_i = inverseRotation * domainScale * rangeScale * noiseDeriv(domainScale * seed);
        derivs += derivs_i;
    }

    return derivs;  
}

//========== END helper functions =========//

// Sum of different frequency noise patterns (fourier series, basically)
float terrainHeight(in vec2 position) {
    return fbm(position, terrain_resolution_factor);
}

vec2 terrainDerivative(in vec2 position) {
    return fbm_deriv(position, terrain_resolution_factor);
}

vec3 terrainNormal(in vec3 position) {
    vec2 terrainDerivs = terrainDerivative(position.xz);

    // Precalculated cross-product of slopes in x and z directions 
    return normalize(vec3(-terrainDerivs.x, 1, -terrainDerivs.y)); 
}

float terrainSDF(in vec3 position) {
    return position.y - terrainHeight(position.xz);
}

// Shadow from other mountains blocking sun. Returns float in range (0, 1).
// Cast a ray from terrainPosition towards sun. Get distance from ray to terrain
// along ray's journey and use to calc shadow.
float terrainShadow(in vec3 terrainPosition) {

    float softnessFactor = 32.0;
    float minT = 0.5;
    float maxT = 10.0; 
    float dt = 0.1;
    // Start at 1, can only get smaller (down to 0) in the following process.
    float minNormalizedTerrainSDF = 1.0;
    
    for (float t = minT; t < maxT; t += dt) {
        vec3 rayPos = rayPosition(terrainPosition, -sun, t);
        float distToTerrain = terrainSDF(rayPos);
        if (distToTerrain <= 0.01) { 
            // Ray intersects terrain, return 0 to indicate total shadow.
            return 0.0;
        }
        // Otherwise, depending on how close ray is to terrain, shadow is not total, but fuzzy.
        float normalizedTerrainSDF = softnessFactor * (distToTerrain / t);
        minNormalizedTerrainSDF = min(minNormalizedTerrainSDF, normalizedTerrainSDF);
    }

    return smoothstep(0.0, 1.0, minNormalizedTerrainSDF);
}

float cloudStrength(vec3 intersectionPosition) {
    return 0.5*smoothstep(-1.0, 8.0, fbm(2.0 * intersectionPosition.xz, 9));
}

float cloudSDF(in vec3 rayOrigin, in vec3 rayDirection) {
    // Treat cloud as plane at y = cloudHeight
    // Equation of ray: ray_O + ray_D*t = (x,y,z)
    // Isolating the y component: ray_O_y + ray_D_y*t = cloudHeight
    // Thus we can easily solve for t, where the plane and ray intersect.
    return (cloudHeight - rayOrigin.y) / rayDirection.y;
}

// Cast ray at shapes in the scene and see what it intersects
// Returns ID of object it hit, if any. Also returns intersection distance as out param.
int castRay(in vec3 rayOrigin, in vec3 rayDirection, out float dist) {
    int ID = -1;
    float maxT = 100.0; // far-clip plane 
    float minT = 0.1;  // near-clip plane
    float dt = 0.1;
    float t, terrainSdfValue, cloudSdfValue;
    float oldT = minT;
    float oldSdfValue = 0.0;
    float intersectionThreshold = 0.0;

    for (t = minT; t < maxT; t += dt) {
        vec3 rayPos = rayPosition(rayOrigin, rayDirection, t);
        
        // Test for intersection with terrain by comparing heights.
        // As t increases, resolution decreseases, so we get less picky about what we consider a "hit"
        terrainSdfValue = terrainSDF(rayPos);
        intersectionThreshold = 0.001*t;
        if (terrainSdfValue <= intersectionThreshold) {
            ID = mountain_targetID;
            // interpolate to give much more accurate results (otherwise shadows are banded and ugly)
            t = oldT + (intersectionThreshold - oldSdfValue) * (t - oldT) / (terrainSdfValue - oldSdfValue); 
            break;
        }
        
        // Increase dt in proportion to t because as we go 
        // further, detail matters less, so take bigger steps.
        dt = max(dt, 0.01*t);

        // Save off old values to help interpolate
        oldT = t;
        oldSdfValue = terrainSdfValue;
    }

    // If we didn't hit anything, we'll default to drawing clouds.
    if (ID == -1) {
        ID = cloud_targetID;
        dist = cloudSDF(rayOrigin, rayDirection);
        return ID;
    }
    
    dist = t;
    return ID;
}

// Real atmosphere haze follows an exponential decay
vec3 addAtmosphere(in vec3 color, in float dist) {
    vec3 haze = exp(atmosphericDecayFactor * dist);      // one exp for each RGB color channel
    return (haze * color) + ((1.0 - haze) * hazeColor); 
}

vec3 skyGradient(in vec3 rayDirection) {
    return skyColor - 0.8*rayDirection.y;
}

vec3 reflectedSkyLight(in float terrainNormal_y) {
    return ((1.0 + terrainNormal_y) / 2.0) * (skyColor / 10.0);
}

// Some sun light is reflected off the ground onto other spots in the ground.
// Take a dot product with the _opposite_ sun direction as used for areas in the sun,
// and use about a 10th of the reflected light to in these spots.
vec3 reflectedGroundLight(in vec3 terrainNormal) {
    return  (dot(terrainNormal, sun) * (landColor / 5.0));
}

// Shift UVs into -1:1 domain and scale to aspect ratio
vec2 scaleUV(in vec2 uv) {
    return (( -1.0 + 2.0 * uv) * vec2(aspectRatio, 1.0));
}

// Create the land material by mixing the land color with snow color where
// the vertical normal is strong enough to support snow.
vec3 landMaterial(float yNormal, float yPos) {
    // Mix land and snow
    float lambda = smoothstep(0.20, 0.25, yNormal);

    return (landColor * (1.0 - lambda) + (landColor * lambda));
}

// Create a coordinate system for a camera placed at the ray cast origin and
// pointed at the given angle
mat3 createCamera(in vec2 angles) {
    vec3 UP = vec3(0.0,1.0,0.0); // global up direction

    vec3 cameraDirection = normalize(vec3(cos(angles.x)*cos(angles.y), -cos(angles.x)*sin(angles.y), cos(angles.y)));
    vec3 cameraRight = normalize(cross(UP, cameraDirection));
    vec3 cameraUp = normalize(cross(cameraDirection, cameraRight));
    
    // This is a change-of-basis matrix. When applied to a vector represented in
    // the space of the ray/target, it transforms the vector to the camera-space.
    return mat3(cameraRight, cameraUp, cameraDirection);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    aspectRatio = iResolution.x/iResolution.y;

    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.xy;
    vec2 scaledUV = scaleUV(uv);
    
    // Cast rays via a camera. The ray and target are defined in global 3D space,
    // The camera is a change-of-basis matrix that takes vectors in camera-space to
    // global-space. In camera-space we aim a ray at each UV coordinate,
    // then transform that ray into its global-space representation and cast it out. 
    vec3 cameraPos = vec3(-5.0, 3.0, 0.0); 
    vec2 cameraAngles = vec2(degreesToRads * 0.0, degreesToRads * 15.0);
    mat3 camera = createCamera(cameraAngles);
    vec3 rayDirection = camera * normalize(vec3(scaledUV, -1.5)); 

    // Cast ray and find point of intersection
    float distIntersect;
    int intersectionID = castRay(cameraPos, rayDirection, distIntersect);
    vec3 intersectionPosition = rayPosition(cameraPos, rayDirection, distIntersect);
    
    // Default color to sky
    vec3 col = skyGradient(rayDirection);
    switch (intersectionID) {
    case mountain_targetID: 
        vec3 normal = terrainNormal(intersectionPosition);
        float sun_shading = -dot(sun, normal);
        float shadow = terrainShadow(intersectionPosition); // shadow from other mountains blocking sun
        vec3 reflectedSkyLight = reflectedSkyLight(normal.y);
        vec3 reflectedGroundLight = reflectedGroundLight(normal);
        vec3 landColor = landMaterial(normal.y, intersectionPosition.y);
        col = (landColor * sun_shading * shadow) + reflectedSkyLight + reflectedGroundLight;
        break;
    case cloud_targetID:
        float cloudMixStrength = cloudStrength(intersectionPosition);
        col = (col * (1.0 - cloudMixStrength)) + (cloudMixStrength * cloudColor);
        // col = vec3(cloudMixStrength);
        break;
    }

   
    // Add atmosphere haze
    col = addAtmosphere(col, distIntersect);
    
    // Smoothstep the color to make dark colors darker and light colors lighter
    // col = smoothstep(0.0, 1.0, col);

    // Output to screen
    fragColor = vec4(col, 1.0);
}