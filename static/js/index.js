window.HELP_IMPROVE_VIDEOJS = false;

function setupScrollToTop() {
  const button = document.querySelector('.scroll-to-top');
  if (!button) return;

  button.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  window.addEventListener('scroll', () => {
    button.classList.toggle('visible', window.scrollY > 300);
  });
}

function setupGalleries() {
  document.querySelectorAll('[data-gallery]').forEach((gallery) => {
    const track = gallery.querySelector('.gallery-track');
    const slides = Array.from(track.querySelectorAll('.gallery-slide'));
    const videos = slides.map((slide) => slide.querySelector('video'));
    const previous = gallery.querySelector('.gallery-prev');
    const next = gallery.querySelector('.gallery-next');
    const position = gallery.querySelector('.gallery-position');
    const dots = gallery.querySelector('.gallery-dots');
    let active = 0;
    let visible = false;

    function syncPlayback() {
      videos.forEach((video, index) => {
        if (visible && index === active) {
          video.play().catch(() => {});
        } else {
          video.pause();
        }
      });
    }

    function setActive(index) {
      if (index === active && dots.children[active]?.getAttribute('aria-current') === 'true') return;
      active = index;
      position.textContent = `${active + 1} / ${slides.length}`;
      previous.disabled = active === 0;
      next.disabled = active === slides.length - 1;
      Array.from(dots.children).forEach((dot, dotIndex) => {
        dot.setAttribute('aria-current', String(dotIndex === active));
      });
      syncPlayback();
    }

    function show(index) {
      const target = Math.max(0, Math.min(slides.length - 1, index));
      track.scrollTo({
        left: target * track.clientWidth,
        behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
      });
    }

    slides.forEach((slide, index) => {
      const dot = document.createElement('button');
      dot.className = 'gallery-dot';
      dot.type = 'button';
      dot.title = slide.querySelector('figcaption').textContent;
      dot.setAttribute('aria-label', `Show ${dot.title}`);
      dot.addEventListener('click', () => show(index));
      dots.appendChild(dot);
    });

    previous.addEventListener('click', () => show(active - 1));
    next.addEventListener('click', () => show(active + 1));
    track.addEventListener('scroll', () => {
      setActive(Math.max(0, Math.min(slides.length - 1, Math.round(track.scrollLeft / track.clientWidth))));
    }, { passive: true });
    track.addEventListener('keydown', (event) => {
      if (event.target !== track || !['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
      event.preventDefault();
      show(active + (event.key === 'ArrowRight' ? 1 : -1));
    });

    setActive(0);
    if ('IntersectionObserver' in window) {
      const observer = new IntersectionObserver(([entry]) => {
        visible = entry.isIntersecting;
        syncPlayback();
      }, { threshold: 0.25 });
      observer.observe(gallery);
    }
  });
}

document.addEventListener('DOMContentLoaded', () => {
  setupScrollToTop();
  setupGalleries();
});
