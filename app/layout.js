import './globals.css';

export const metadata = {
  title: 'CardioAI | AI-Based ECG Heart Anomaly Detection & Research Dashboard',
  description: 'Academic biomedical research dashboard for automated arrhythmia classification and heart anomaly detection using the MIT-BIH Arrhythmia Database.',
  keywords: 'ECG, Arrhythmia, MIT-BIH, AAMI EC57, Deep Learning, 1D-CNN, Cardiology, Machine Learning',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
