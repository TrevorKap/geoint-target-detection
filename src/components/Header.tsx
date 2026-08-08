interface HeaderProps {
  status: 'idle' | 'loaded' | 'analyzing';
}

const STATUS_LABEL: Record<HeaderProps['status'], string> = {
  idle: 'AWAITING RASTER',
  loaded: 'RASTER STAGED',
  analyzing: 'INFERENCE RUNNING',
};

export default function Header({ status }: HeaderProps) {
  return (
    <header className="app-header">
      <div className="app-header__brand">
        <span className="app-header__glyph">🛰</span>
        <div>
          <h1 className="app-header__title">TACTICAL GEOINT ANALYZER</h1>
          <p className="app-header__subtitle">
            Overhead Target Segmentation · Sliding-Window Inference · GIS Vector Export
          </p>
        </div>
      </div>
      <div className={`app-header__status app-header__status--${status}`}>
        <span className="app-header__status-dot" />
        {STATUS_LABEL[status]}
      </div>
    </header>
  );
}
