import styles from './Info.module.css';
import ecgImage from '../../assets/ecg-derivations.png';
import stImage from '../../assets/st-elevation.png';
import smartwatchImage from '../../assets/smartwatch-ecg.png';
import aiImage from '../../assets/ai-recognition.png';

export default function Info() {
  return (
    <section className={styles.infoSection}>
      <h2 className={styles.title}>How it works</h2>

      {/* Sección 1: Derivaciones del ECG */}
      <div className={styles.card}>
        <div className={styles.text}>
          <h3>12-Lead ECG</h3>
          <p>
            Traditional electrocardiograms use 12 leads measured by 10 electrodes placed on the body.
            These allow doctors to analyze the electrical activity of the heart in detail.
          </p>
        </div>
        <div className={styles.image}>
         <img src={ecgImage} alt="12-lead ECG derivations" />
        </div>
      </div>

      {/* Sección 2: Anomalías detectables */}
      <div className={styles.card}>
        <div className={styles.text}>
          <h3>Detecting Anomalies</h3>
          <p>
            Abnormalities like ST-segment elevation can indicate myocardial infarction or other cardiac issues.
            ECGs allow early detection of these signs.
          </p>
        </div>
        <div className={styles.image}>
          <img src={stImage} alt="ST-segment elevation" />
        </div>
      </div>

      {/* Sección 3: Smartwatch limitations */}
      <div className={styles.card}>
        <div className={styles.text}>
          <h3>Smartwatch ECG</h3>
          <p>
            Smartwatches typically measure only one lead (homologous to D-II). 
            Although limited, they can still detect certain anomalies and signs of cardiovascular disease or arrhythmias.
          </p>
        </div>
        <div className={styles.image}>
          <img src={smartwatchImage} alt="Smartwatch ECG" />
        </div>
      </div>

      {/* Sección 4: AI recognition */}
      <div className={styles.card}>
        <div className={styles.text}>
          <h3>AI-Powered Analysis</h3>
          <p>
            Artificial intelligence analyzes ECG signals using image recognition techniques,
            allowing automated detection of arrhythmias and other heart conditions.
          </p>
        </div>
        <div className={styles.image}>
          <img src={aiImage} alt="AI ECG recognition" />
        </div>
      </div>
    </section>
  );
}
