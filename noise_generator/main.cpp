#include <random>
#include <array>
#include <iostream>
#include "TrainManager.hpp"

namespace raisim {

extern "C"
int main(int argc, char * argv[]) {
  static std::mt19937 gen;
  static std::uniform_real_distribution<double> uniDist(0., 1.);
  static std::normal_distribution<double> normDist(0., 1.);

  HeightNoiseGenerator heightNoiseGenerator;
  HeightNoiseGenerator::Noise heightNoise;

  // initialize container of the true height scan data
  const int nFoots = 4;
  const int nScansPerFoot = 10;
  std::array<std::array<double, nScansPerFoot>, nFoots> heightScan;
  std::array<std::array<double, nScansPerFoot>, nFoots> heightScanNoisy;
  for (int i = 0; i < nFoots; i++)
    for (int j = 0; j < nScansPerFoot; j++)
      heightScan[i][j] = 0;

  // call when the episode starts
  heightNoiseGenerator.sampleNoiseType(gen, uniDist);
  heightNoiseGenerator.sampleNoise(
    heightNoise, HeightNoiseGenerator::SampleType::INIT, gen, uniDist, normDist);

  // simulate
  int nSteps = 400;
  for (int t = 0; t < nSteps; t++) {
    for (int i = 0; i < nFoots; i++) {
      // call when the foot changes
      heightNoiseGenerator.sampleNoise(
        heightNoise, HeightNoiseGenerator::SampleType::FOOT_CHANGE, gen, uniDist, normDist);

      for (int j = 0; j < nScansPerFoot; j++) {
        // call for every height scan points
        heightNoiseGenerator.sampleNoise(
          heightNoise, HeightNoiseGenerator::SampleType::POINT_CHANGE, gen, uniDist, normDist);

        double xOffset = heightNoise.x;
        double yOffset = heightNoise.y;
        double zOffset = heightNoise.z;

        /// Read height scan with x, y, z offset from the true value as below.
        /// heightScanNoisy[i][j] = heightmap->getHeight(default_x_ij + xOffset, default_y_ij + yOffset) + zOffset
        /// However, in this code, there is no heightmap generated.
        /// Thus, only zOffset is added to the default height value to show the example use case.
        heightScanNoisy[i][j] = heightScan[i][j] + zOffset;

        std::cout << xOffset << " " << yOffset << " " << zOffset << "\n";
      }
    }
  }

  return 0;
}

}