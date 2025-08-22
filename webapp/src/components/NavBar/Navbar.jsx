import styles from "./Navbar.module.css";

export default function Navbar(){
return(
    <>
<div className={styles.navbar}>
    <h1 className={styles.logo}>CardioAI</h1>
      <ul className={styles.links}>
        <li><a href="#">Home</a></li>
        <li><a href="#">About Us</a></li>
        <li><a href="#">Contact</a></li>
      </ul>
</div>
    </>
)
}