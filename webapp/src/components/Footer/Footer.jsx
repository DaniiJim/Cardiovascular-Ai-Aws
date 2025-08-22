import styles from './Footer.module.css';

export default function Footer() {
  return (
    <footer className={styles.footer}>
      <p>© 2025 CardioAI. All rights reserved.</p>
      <div className={styles.social}>
        <a href="#" aria-label="Facebook">GitHub</a>
        
      </div>
    </footer>
  );
}
