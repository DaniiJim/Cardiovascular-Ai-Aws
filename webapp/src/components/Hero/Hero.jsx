import styles from './Hero.module.css';
import heroImage from '../../assets/hero-image.png'; 

export default function Hero() {
  return (
    <section className={styles.hero}>
      <div className={styles.content}>
        <h1 className={styles.title}>
          WHERE HEALTH <br /> MEETS TECHNOLOGY
        </h1>
        <p className={styles.subtitle}>
          Take care of your heart from anywhere
        </p>
        <button className={styles.cta}>Try it now</button>
      </div>
      <div className={styles.imageContainer}>
        <img src={heroImage} alt="Heart + Electro + Smartwatch" />
      </div>
    </section>
  );
}
