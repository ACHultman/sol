import * as tf from "@tensorflow/tfjs";
import data from "../planet-data.json";

const numberOfPlanets = data.planets.length;

const xInit = data.planets.map((planet) => planet.x);
const yInit = data.planets.map((planet) => planet.y);
const masses = data.planets.map((planet) => planet.m);

const xInitTensor = tf.tensor2d(xInit, [numberOfPlanets, 3]);
const yInitTensor = tf.tensor2d(yInit, [numberOfPlanets, 3]);
const G = tf.scalar(data.G);

export const Sol = () => {
  return <div></div>;
};

export default Sol;
