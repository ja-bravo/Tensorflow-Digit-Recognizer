import Head from 'next/head';
import styles from '../styles/Home.module.css';
import { MnistData } from '../mnist';
import { useCallback, useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

export default function Home() {
  const dataRef = useRef<MnistData>();
  const modelRef = useRef<tf.Sequential>();
  const [testData, setTestData] = useState<{ xs: tf.Tensor4D; labels: tf.Tensor2D }>();
  const [trainData, setTrainData] = useState<{ xs: tf.Tensor4D; labels: tf.Tensor2D }>();

  useEffect(() => {
    console.log('loading');
    dataRef.current = new MnistData();
    dataRef.current.load().then(r => {
      setTestData(dataRef.current.getTestData(undefined));
      setTrainData(dataRef.current.getTrainData());
    });
  }, []);

  useEffect(() => {
    if (testData && trainData) {
      const m = tf.sequential();
      m.add(
        tf.layers.conv2d({
          activation: 'relu',
          inputShape: [28, 28, 1],
          kernelSize: 3,
          filters: 16,
        }),
      );
      m.add(
        tf.layers.maxPooling2d({
          poolSize: 2,
          strides: 2,
        }),
      );
      m.add(tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: 'relu' }));
      // m.add(tf.layers.batchNormalization());
      // m.add(tf.layers.dropout({ rate: 0.25 }));
      m.add(
        tf.layers.maxPooling2d({
          poolSize: 2,
          strides: 2,
        }),
      );
      m.add(tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: 'relu' }));
      // m.add(tf.layers.dropout({ rate: 0.25 }));
      m.add(tf.layers.flatten({}));
      m.add(tf.layers.dense({ units: 64, activation: 'relu' }));
      m.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

      m.compile({
        optimizer: 'rmsprop',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
      });
      modelRef.current = m;
      const surface = { name: 'Model Summary', tab: 'Model Inspection' };
      tfvis.show.modelSummary(surface, m);
      console.log('model created and compiled');
    }
  }, [testData, trainData]);

  const trainModel = useCallback(async () => {
    const surface = { name: 'Callbacks', tab: 'Training' };
    await modelRef.current.fit(trainData.xs, trainData.labels, {
      callbacks: tfvis.show.fitCallbacks(surface, ['loss', 'acc']),
      epochs: 5,
      batchSize: 512,
      validationSplit: 0.15,
      shuffle: true,
    });

    const evRes = await modelRef.current.evaluate(testData.xs, testData.labels);
    const accuracy = evRes[1].dataSync()[0] * 100;
    console.log(`Test accuracy: ${accuracy.toFixed(1)}%`);

    const preds = (modelRef.current.predict(testData.xs) as tf.Tensor).argMax(-1).as1D();
    const labels = testData.labels.argMax(-1).as1D();
    const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
    const container = { name: 'Accuracy', tab: 'Evaluation' };
    await tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

    const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
    const container2 = { name: 'Confusion Matrix', tab: 'Evaluation' };
    await tfvis.render.confusionMatrix(container2, { values: confusionMatrix, tickLabels: classNames });

    await modelRef.current.save('downloads://model');
    modelRef.current.dispose();
  }, [modelRef.current, trainData]);

  return (
    <div className={styles.container}>
      <Head>
        <title>Create Next App</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className={styles.main}>
        <h1 className={styles.title}>
          Welcome to <a href="https://nextjs.org">Next.js!</a>
        </h1>

        <button onClick={() => trainModel()}>Train Model</button>
      </main>
    </div>
  );
}
