import React, { useCallback, useMemo, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import data from "../planet-data.json";

const numOfPlanets = data.planets.length;

const numberOfPlanets = data.planets.length;

const xPosInitialArray = data.planets.map((planet) => planet.x);
const velocityInitialArray = data.planets.map((planet) => planet.v);
const masses = data.planets.map((planet) => planet.m);
const xPosInitial = tf.tensor2d(xPosInitialArray, [numberOfPlanets, 3]);
const velocityInitial = tf.tensor2d(velocityInitialArray, [numberOfPlanets, 3]);
const G = tf.scalar(data.G);

export const Sol = ({ dt = 0.1 }) => {
  const [pos, setPos] = useState(xPosInitialArray);
  // useRef to prevent re-render on change
  const xPos = useRef(xPosInitial);
  const velocity = useRef(velocityInitial);
  // create dtTensor from dt
  const dtTensor = useMemo(() => tf.scalar(dt), [dt]);
  const compute = useCallback(() => {
    // release from memory
    const [newXPos, newVelocity] = tf.tidy(() => {
      const acceleration = calcAcceleration(xPosInitial);
      const newXPos = xPos.current.add(tf.mul(velocity.current, dtTensor));
      const newVelocity = velocity.current.add(tf.mul(acceleration, dtTensor));

      return [newXPos, newVelocity];
    });

    // release memory of old values
    tf.dispose(xPos.current, velocity);

    // update x postiion and velocity
    xPos.current = newXPos;
    velocity.current = newVelocity;

    newXPos.array().then((newPos) => {
      setPos(newPos);
    });
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
