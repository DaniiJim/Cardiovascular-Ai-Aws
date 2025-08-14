import React, { useState, useEffect } from 'react';
import bocaAbierta from '../../assets/boca_abierta.jpg';
import bocaCerrada from '../../assets/boca_cerrada.jpg';
import './DoctorImage.css';

const DoctorImage = () => {
  const [isMouthOpen, setIsMouthOpen] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      setIsMouthOpen(prevState => !prevState);
    }, 500); // Cambia cada medio segundo para un efecto más natural

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="doctor-image-wrapper">
      <img src={isMouthOpen ? bocaAbierta : bocaCerrada} alt="Doctor" />
    </div>
  );
};

export default DoctorImage;
