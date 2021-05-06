import React, { useCallback } from "react";
import * as tf from "@tensorflow/tfjs";
import data from "../planet-data.json";

const numOfPlanets = data.planets.length;

const numberOfPlanets = data.planets.length;

const xInitialArray = data.planets.map((planet) => planet.x);
const vInitialArray = data.planets.map((planet) => planet.v);
const masses = data.planets.map((planet) => planet.m);
const xInitial = tf.tensor2d(xInitialArray, [numberOfPlanets, 3]);
const vInitial = tf.tensor2d(vInitialArray, [numberOfPlanets, 3]);
const G = tf.scalar(data.G);

export const Sol = () => {
  const compute = useCallback(() => {
    const a = calcAcceleration(xInitial);
    a.print();
  });
  compute();
  return <div></div>;
};

const calcAcceleration = (x) => {
  const unstackedX = tf.unstack(x);
  const accelerations = Array(numberOfPlanets).fill(tf.tensor1d([0, 0, 0]));

  for (let i = 0; i < numOfPlanets; i++) {
    const iX = unstackedX[i];
    // loop through remaining planets
    for (let j = i + 1; j < numOfPlanets; j++) {
      const jX = unstackedX[j];
      // get vector by subtracting jx from ix
      const vector = tf.sub(jX, iX);
      // get radius by normalizing vector
      const radius = tf.norm(vector);
      // gravitational force calculation
      const gravForce = G.mul(masses[i])
        .mul(masses[j])
        .div(tf.pow(radius, 3))
        .mul(vector);

      // adjust forces for each
      accelerations[i] = accelerations[i].add(gravForce);
      accelerations[j] = accelerations[j].sub(gravForce);
    }

    // calculate acceleration for planet i
    accelerations[i] = accelerations[i].div(masses[i]);
  }

  return tf.stack(accelerations);
};

export default Sol;
