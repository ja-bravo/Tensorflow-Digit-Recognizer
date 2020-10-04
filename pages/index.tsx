import Head from 'next/head';
import styles from '../styles/Home.module.css';
import { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import CanvasDraw from 'react-canvas-draw';

const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];
export default function Home() {
  const modelRef = useRef<tf.LayersModel>();
  const canvasRef = useRef<CanvasDraw>();
  const [modelLoading, setModelLoading] = useState(true);
  const [probs, setProbs] = useState<{ prob: number; index: number }[]>();
  const [answer, setAnswer] = useState<number>();

  useEffect(() => {
    tf.loadLayersModel('/model.json').then(r => {
      modelRef.current = r;
      setModelLoading(false);
      console.log(r.summary());
    });
  }, []);

  return (
    <div className={styles.container}>
      <Head>
        <title>Create Next App</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className={styles.main}>
        <h1 className={styles.title}>{modelLoading ? 'Loading model...' : 'Model Ready'}</h1>
        {probs && (
          <p>
            {probs.map(p => (
              <>
                <span style={{ fontWeight: 'bold' }}>{p.index}:</span>
                <span style={{ marginRight: 8 }}>{Math.round(p.prob * 100)}%</span>
              </>
            ))}
          </p>
        )}
        {answer !== undefined && <p>Prediction: {answer}</p>}

        {!modelLoading && (
          <>
            <div style={{ display: 'flex', flexDirection: 'row', justifyContent: 'center' }}>
              <CanvasDraw
                brushColor={'white'}
                style={{ background: 'black', marginRight: 32 }}
                ref={r => (canvasRef.current = r!)}
                canvasWidth={504}
                canvasHeight={504}
                brushRadius={10}
                lazyRadius={0}
                onChange={async e => {
                  const answer = tf.tidy(() => {
                    const tensor = tf.browser.fromPixels(e.canvas.drawing as HTMLCanvasElement);
                    const resized = tf.image.resizeBilinear(tensor, [28, 28], true);
                    const oneChannel = resized.slice([0, 0, 0], [resized.shape[0], resized.shape[1], 1]);
                    tf.browser.toPixels(oneChannel.asType('int32'), document.getElementById('print') as any);
                    const res = modelRef.current.predict(oneChannel.expandDims(0)) as tf.Tensor;
                    canvasRef.current.clear();

                    return res;
                  });

                  const probs = Array.from(answer.dataSync());
                  setProbs(probs.map((p, i) => ({ prob: p, index: i })));
                  setAnswer(classNames[answer.argMax(-1).dataSync()[0]]);
                  answer.dispose();
                }}
              />

              <div>
                <p>Input to the model</p>
                <canvas id="print" style={{ width: 150, height: 150 }} />
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
