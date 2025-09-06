import React, { useEffect, useState } from 'react';
import bocaAbierta from '../../assets/boca_abierta.jpg';
import bocaCerrada from '../../assets/boca_cerrada.jpg';
import './DoctorImage.css';


const DoctorImage = () => {
const [open, setOpen] = useState(false);
useEffect(() => {
const id = setInterval(() => setOpen(v => !v), 900);
return () => clearInterval(id);
}, []);
return (
<div className="doctor-frame">
<img src={open ? bocaAbierta : bocaCerrada} alt="Doctor" />
<div className="ai-pill">AI</div>
</div>
);
};


export default DoctorImage;