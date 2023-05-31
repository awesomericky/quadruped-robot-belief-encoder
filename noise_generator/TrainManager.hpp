#ifndef TRAIN_MANAGER_HPP
#define TRAIN_MANAGER_HPP

namespace raisim {

class HeightNoiseGenerator {
  public:
    enum class NoiseType : int {
      NOMINAL = 0,
      OFFSET,
      NOISY
    };

    // INIT (episode start) --> FOOT_CHANGE (foot change) --> POINT_CHANGE (point change)
    enum class SampleType : int {
      INIT = 0,
      FOOT_CHANGE,
      POINT_CHANGE
    };

    struct Noise {
      double x;
      double y;
      double z;
    };

    struct NoiseParam {
      Noise pointAndTime;  // sampled for each point every time step
      Noise footAndTime;  // sampled for each foot every time step
      Noise foot;  // sampled for each foot at the beginning of the episode
      double zOutlier;
      double constantOffsetProb;
      double zOutlierProb;
    };

    HeightNoiseGenerator() = default;

    ~HeightNoiseGenerator() = default;

    void sampleNoiseType(std::mt19937 &gen,
                         std::uniform_real_distribution<double> &uniDist) {
      double val = uniDist(gen);

      if (val < 0.6)
        noiseType = NoiseType::NOMINAL;
      else if (val < 0.6 + 0.3)
        noiseType = NoiseType::OFFSET;
      else
        noiseType = NoiseType::NOISY;

      switch (noiseType) {
        case NoiseType::NOMINAL:
          defaultNoiseParam.pointAndTime.x = 0.004;
          defaultNoiseParam.pointAndTime.y = 0.004;
          defaultNoiseParam.pointAndTime.z = 0.005;
          defaultNoiseParam.footAndTime.x = 0.01;
          defaultNoiseParam.footAndTime.y = 0.01;
          defaultNoiseParam.footAndTime.z = 0.04;
          defaultNoiseParam.zOutlierProb = 0.02;
          defaultNoiseParam.zOutlier = 0.03;
          defaultNoiseParam.constantOffsetProb = 0.05;
          defaultNoiseParam.foot.x = 0.1;
          defaultNoiseParam.foot.y = 0.1;
          defaultNoiseParam.foot.z = 0.1;
          break;
        case NoiseType::OFFSET:
          defaultNoiseParam.pointAndTime.x = 0.004;
          defaultNoiseParam.pointAndTime.y = 0.004;
          defaultNoiseParam.pointAndTime.z = 0.005;
          defaultNoiseParam.footAndTime.x = 0.01;
          defaultNoiseParam.footAndTime.y = 0.01;
          defaultNoiseParam.footAndTime.z = 0.1;
          defaultNoiseParam.zOutlierProb = 0.02;
          defaultNoiseParam.zOutlier = 0.1;
          defaultNoiseParam.constantOffsetProb = 0.02;
          defaultNoiseParam.foot.x = 0.1;
          defaultNoiseParam.foot.y = 0.1;
          defaultNoiseParam.foot.z = 0.1;
          break;
        case NoiseType::NOISY:
          defaultNoiseParam.pointAndTime.x = 0.004;
          defaultNoiseParam.pointAndTime.y = 0.004;
          defaultNoiseParam.pointAndTime.z = 0.1;
          defaultNoiseParam.footAndTime.x = 0.1;
          defaultNoiseParam.footAndTime.y = 0.1;
          defaultNoiseParam.footAndTime.z = 0.3;
          defaultNoiseParam.zOutlierProb = 0.05;
          defaultNoiseParam.zOutlier = 0.3;
          defaultNoiseParam.constantOffsetProb = 0.3;
          defaultNoiseParam.foot.x = 0.1;
          defaultNoiseParam.foot.y = 0.1;
          defaultNoiseParam.foot.z = 0.1;
          break;
      }
    }

    void sampleNoise(Noise &noise,
                     const SampleType &sampleType,
                     std::mt19937 &gen,
                     std::uniform_real_distribution<double> &uniDist,
                     std::normal_distribution<double> &normDist) {
      switch (sampleType) {
        case SampleType::INIT:
          if (uniDist(gen) < defaultNoiseParam.constantOffsetProb) {
            sampledNoiseParam.foot.x = defaultNoiseParam.foot.x * normDist(gen);
            sampledNoiseParam.foot.y = defaultNoiseParam.foot.y * normDist(gen);
            sampledNoiseParam.foot.z = defaultNoiseParam.foot.z * normDist(gen);
          } else {
            sampledNoiseParam.foot.x = 0.;
            sampledNoiseParam.foot.y = 0.;
            sampledNoiseParam.foot.z = 0.;
          }
          return;
        case SampleType::FOOT_CHANGE:
          sampledNoiseParam.footAndTime.x = defaultNoiseParam.footAndTime.x * normDist(gen);
          sampledNoiseParam.footAndTime.y = defaultNoiseParam.footAndTime.y * normDist(gen);
          sampledNoiseParam.footAndTime.z = defaultNoiseParam.footAndTime.z * normDist(gen);
          return;
        case SampleType::POINT_CHANGE:
          sampledNoiseParam.pointAndTime.x = defaultNoiseParam.pointAndTime.x * normDist(gen);
          sampledNoiseParam.pointAndTime.y = defaultNoiseParam.pointAndTime.y * normDist(gen);
          sampledNoiseParam.pointAndTime.z = defaultNoiseParam.pointAndTime.z * normDist(gen);
          if (uniDist(gen) < defaultNoiseParam.zOutlierProb)
            sampledNoiseParam.zOutlier = defaultNoiseParam.zOutlier * normDist(gen);
          else
            sampledNoiseParam.zOutlier = 0.;
          break;
      }

      noise.x = sampledNoiseParam.foot.x + sampledNoiseParam.footAndTime.x + sampledNoiseParam.pointAndTime.x;
      noise.y = sampledNoiseParam.foot.y + sampledNoiseParam.footAndTime.y + sampledNoiseParam.pointAndTime.y;
      noise.z = sampledNoiseParam.foot.z + sampledNoiseParam.footAndTime.z +
                sampledNoiseParam.pointAndTime.z + sampledNoiseParam.zOutlier;
    }

  private:
    NoiseType noiseType;
    NoiseParam defaultNoiseParam, sampledNoiseParam;
};

}  // namespace raisim

#endif    // TRAIN_MANAGER_HPP