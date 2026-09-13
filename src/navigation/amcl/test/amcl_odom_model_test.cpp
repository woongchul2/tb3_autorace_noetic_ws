/*
 * Regression tests for the signed sub-centimetre differential odom model.
 *
 * Copyright (C) 2026 Custom AutoRace contributors
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include <cmath>
#include <initializer_list>

#include <gtest/gtest.h>

#include "amcl/pf/pf.h"
#include "amcl/sensors/amcl_odom.h"

namespace
{

pf_vector_t Pose(double x, double y, double yaw)
{
  pf_vector_t pose = pf_vector_zero();
  pose.v[0] = x;
  pose.v[1] = y;
  pose.v[2] = yaw;
  return pose;
}

double AngleDifference(double first, double second)
{
  return std::atan2(
      std::sin(first - second),
      std::cos(first - second));
}

pf_vector_t ApplySingleParticleUpdate(
    amcl::odom_model_t model,
    const pf_vector_t& particle_pose,
    const pf_vector_t& old_odom_pose,
    const pf_vector_t& odom_delta)
{
  pf_t* filter = pf_alloc(1, 1, 0.0, 0.0, nullptr, nullptr);
  pf_sample_t* sample =
      &filter->sets[filter->current_set].samples[0];
  sample->pose = particle_pose;

  amcl::AMCLOdom odom;
  odom.SetModel(model, 0.0, 0.0, 0.0, 0.0);
  amcl::AMCLOdomData data;
  data.delta = odom_delta;
  data.pose = pf_vector_add(old_odom_pose, odom_delta);
  EXPECT_TRUE(odom.UpdateAction(filter, &data));

  const pf_vector_t result = sample->pose;
  pf_free(filter);
  return result;
}

TEST(AmclOdomModel, StockDiffReproducesShortReverseSignLoss)
{
  const pf_vector_t result = ApplySingleParticleUpdate(
      amcl::ODOM_MODEL_DIFF,
      Pose(0.0, 0.0, 0.0),
      Pose(0.0, 0.0, 0.0),
      Pose(-0.005, 0.0, 0.0));

  EXPECT_NEAR(result.v[0], 0.005, 1e-12);
  EXPECT_NEAR(result.v[1], 0.0, 1e-12);
  EXPECT_NEAR(result.v[2], 0.0, 1e-12);
}

TEST(AmclOdomModel, SignedDiffPreservesShortReverseTravel)
{
  const pf_vector_t result = ApplySingleParticleUpdate(
      amcl::ODOM_MODEL_DIFF_SIGNED,
      Pose(0.0, 0.0, 0.0),
      Pose(0.0, 0.0, 0.0),
      Pose(-0.005, 0.0, 0.0));

  EXPECT_NEAR(result.v[0], -0.005, 1e-12);
  EXPECT_NEAR(result.v[1], 0.0, 1e-12);
  EXPECT_NEAR(result.v[2], 0.0, 1e-12);
}

TEST(AmclOdomModel, SignedDiffKeepsShortForwardTravelUnchanged)
{
  const pf_vector_t result = ApplySingleParticleUpdate(
      amcl::ODOM_MODEL_DIFF_SIGNED,
      Pose(0.0, 0.0, 0.0),
      Pose(0.0, 0.0, 0.0),
      Pose(0.005, 0.0, 0.0));

  EXPECT_NEAR(result.v[0], 0.005, 1e-12);
  EXPECT_NEAR(result.v[1], 0.0, 1e-12);
  EXPECT_NEAR(result.v[2], 0.0, 1e-12);
}

TEST(AmclOdomModel, SignedDiffUsesTheOldOdomHeadingForReverseSign)
{
  const double particle_yaw = 0.4;
  const pf_vector_t result = ApplySingleParticleUpdate(
      amcl::ODOM_MODEL_DIFF_SIGNED,
      Pose(0.0, 0.0, particle_yaw),
      Pose(1.0, 2.0, M_PI_2),
      Pose(0.0, -0.006, 0.0));

  EXPECT_NEAR(result.v[0], -0.006 * std::cos(particle_yaw), 1e-12);
  EXPECT_NEAR(result.v[1], -0.006 * std::sin(particle_yaw), 1e-12);
  EXPECT_NEAR(result.v[2], particle_yaw, 1e-12);
}

TEST(AmclOdomModel, SignedDiffLeavesPureRotationUnchanged)
{
  const pf_vector_t result = ApplySingleParticleUpdate(
      amcl::ODOM_MODEL_DIFF_SIGNED,
      Pose(0.2, -0.3, 0.4),
      Pose(1.0, 2.0, -0.7),
      Pose(0.0, 0.0, 0.08));

  EXPECT_NEAR(result.v[0], 0.2, 1e-12);
  EXPECT_NEAR(result.v[1], -0.3, 1e-12);
  EXPECT_NEAR(result.v[2], 0.48, 1e-12);
}

TEST(AmclOdomModel, SignedDiffKeepsNormalReverseTravelEquivalentToStock)
{
  const pf_vector_t particle = Pose(0.3, -0.2, 0.4);
  const pf_vector_t old_odom = Pose(1.0, 2.0, -0.6);
  const pf_vector_t delta = Pose(-0.04, 0.03, 0.12);
  const pf_vector_t stock = ApplySingleParticleUpdate(
      amcl::ODOM_MODEL_DIFF, particle, old_odom, delta);
  const pf_vector_t signed_model = ApplySingleParticleUpdate(
      amcl::ODOM_MODEL_DIFF_SIGNED, particle, old_odom, delta);

  EXPECT_NEAR(signed_model.v[0], stock.v[0], 1e-12);
  EXPECT_NEAR(signed_model.v[1], stock.v[1], 1e-12);
  EXPECT_NEAR(signed_model.v[2], stock.v[2], 1e-12);
}

TEST(AmclOdomModel, SignedDiffPreservesReverseSignAcrossGuardBoundary)
{
  for (const double distance : {0.009999, 0.010000, 0.010001})
  {
    const pf_vector_t result = ApplySingleParticleUpdate(
        amcl::ODOM_MODEL_DIFF_SIGNED,
        Pose(0.0, 0.0, 0.0),
        Pose(0.0, 0.0, 0.0),
        Pose(-distance, 0.0, 0.0));

    EXPECT_NEAR(result.v[0], -distance, 1e-12);
    EXPECT_NEAR(result.v[1], 0.0, 1e-12);
    EXPECT_NEAR(AngleDifference(result.v[2], 0.0), 0.0, 1e-12);
  }
}

TEST(AmclOdomModel, SignedDiffKeepsSubCentimetreReverseArcInReverseQuadrant)
{
  const double radius = 0.114;
  const int update_count = 25;
  const double yaw_step = M_PI_2 / update_count;
  pf_vector_t odom_pose = Pose(0.0, 0.0, 0.0);
  pf_vector_t particle_pose = Pose(0.0, 0.0, 0.0);

  for (int index = 0; index < update_count; ++index)
  {
    const double next_yaw = odom_pose.v[2] + yaw_step;
    const pf_vector_t next_odom_pose = Pose(
        odom_pose.v[0] - radius *
            (std::sin(next_yaw) - std::sin(odom_pose.v[2])),
        odom_pose.v[1] + radius *
            (std::cos(next_yaw) - std::cos(odom_pose.v[2])),
        next_yaw);
    const pf_vector_t delta = pf_vector_sub(next_odom_pose, odom_pose);
    ASSERT_LT(std::hypot(delta.v[0], delta.v[1]), 0.01);
    particle_pose = ApplySingleParticleUpdate(
        amcl::ODOM_MODEL_DIFF_SIGNED,
        particle_pose,
        odom_pose,
        delta);
    odom_pose = next_odom_pose;
  }

  EXPECT_LT(particle_pose.v[0], 0.0);
  EXPECT_LT(particle_pose.v[1], 0.0);
  EXPECT_LT(
      std::hypot(
          particle_pose.v[0] + radius,
          particle_pose.v[1] + radius),
      0.006);
  EXPECT_NEAR(particle_pose.v[2], M_PI_2, 1e-12);
}

}  // namespace

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
