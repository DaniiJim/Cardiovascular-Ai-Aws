import { useState } from 'react';
import styles from './Analyze.module.css';

export default function Analyze() {
  const [image, setImage] = useState(null);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(URL.createObjectURL(file));
    }
  };

  return (
    <section className={styles.analyzeSection}>
      <h2 className={styles.sectionTitle}>Try it now</h2>
      <div className={styles.analyzeContainer}>
        <div className={styles.uploadContainer}>
          <label className={styles.uploadButton}>
            Upload Image
            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              hidden
            />
          </label>

          {image ? (
            <div className={styles.preview}>
              <img src={image} alt="Uploaded ECG" />
            </div>
          ) : (
            <div className={styles.previewPlaceholder}>
              Image preview will appear here
            </div>
          )}
        </div>

        <div className={styles.resultsContainer}>
          <h3>Analysis Results</h3>
          <div className={styles.resultsPlaceholder}>
            Results will appear here
          </div>
        </div>
      </div>
    </section>
  );
}
