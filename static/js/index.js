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

function setupCodeTracking() {
  const codeLink = document.querySelector('[data-code-link]');
  if (!codeLink) return;

  codeLink.addEventListener('click', () => {
    if (typeof window.gtag === 'function') {
      window.gtag('event', 'code_click', { link_url: codeLink.href });
    }
  });
}

function setupGalleries() {
  document.querySelectorAll('[data-gallery]').forEach((gallery) => {
    const track = gallery.querySelector('.gallery-track');
    const slides = Array.from(track.querySelectorAll('.gallery-slide'));
    const videos = slides.map((slide) => slide.querySelector('video'));
    const count = slides.length;
    const previous = gallery.querySelector('.gallery-prev');
    const next = gallery.querySelector('.gallery-next');
    const previewPrevious = gallery.querySelector('.gallery-peek-prev');
    const previewNext = gallery.querySelector('.gallery-peek-next');
    const position = gallery.querySelector('.gallery-position');
    const dots = gallery.querySelector('.gallery-dots');
    let active = 0;
    let visible = false;
    let transitioning = false;
    let queued;
    let touchStart;

    function wrap(index) {
      return (index + count) % count;
    }

    function syncPlayback() {
      videos.forEach((video, index) => {
        if (visible && !transitioning && index === active) {
          if (video.readyState === 0) video.load();
          video.play().catch(() => {});
        } else {
          video.pause();
        }
      });
    }

    function updatePreview(button, index) {
      const target = wrap(index);
      const label = slides[target].querySelector('figcaption').textContent;
      button.hidden = false;
      button.title = `Show ${label}`;
      button.setAttribute('aria-label', button.title);
      button.querySelector('img').src = videos[target].poster;
    }

    function setActive(index) {
      active = wrap(index);
      position.textContent = `${active + 1} / ${count}`;
      Array.from(dots.children).forEach((dot, dotIndex) => {
        dot.setAttribute('aria-current', String(dotIndex === active));
      });
      updatePreview(previewPrevious, active - 1);
      updatePreview(previewNext, active + 1);
    }

    function show(index, direction) {
      const target = wrap(index);
      if (transitioning) {
        queued = { index: target, direction };
        return;
      }
      if (target === active) return;

      const forward = (target - active + count) % count;
      const side = direction || (forward <= count / 2 ? 1 : -1);
      const source = active;
      const outgoing = slides[source];
      const incoming = slides[target];
      transitioning = true;
      syncPlayback();
      videos[source].controls = false;
      videos[target].controls = false;
      outgoing.classList.remove('is-active');
      outgoing.classList.add('is-exiting', side > 0 ? 'to-left' : 'to-right');
      outgoing.setAttribute('aria-hidden', 'true');
      outgoing.inert = true;
      incoming.classList.add('is-entering', side > 0 ? 'from-right' : 'from-left');
      incoming.removeAttribute('aria-hidden');
      incoming.inert = false;
      setActive(target);

      function finish() {
        videos[source].controls = true;
        videos[target].controls = true;
        outgoing.classList.remove('is-exiting', 'to-left', 'to-right');
        incoming.classList.remove('is-entering', 'from-left', 'from-right');
        incoming.classList.add('is-active');
        transitioning = false;
        syncPlayback();
        if (queued) {
          const nextRequest = queued;
          queued = undefined;
          show(nextRequest.index, nextRequest.direction);
        }
      }

      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        finish();
      } else {
        incoming.addEventListener('animationend', finish, { once: true });
      }
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

    previous.addEventListener('click', () => show(active - 1, -1));
    next.addEventListener('click', () => show(active + 1, 1));
    previewPrevious.addEventListener('click', () => show(active - 1, -1));
    previewNext.addEventListener('click', () => show(active + 1, 1));
    videos.forEach((video, index) => {
      video.addEventListener('canplay', () => {
        if (index === active) syncPlayback();
      });
    });
    track.addEventListener('keydown', (event) => {
      if (event.target !== track || !['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
      event.preventDefault();
      const direction = event.key === 'ArrowRight' ? 1 : -1;
      show(active + direction, direction);
    });
    track.addEventListener('touchstart', (event) => {
      touchStart = { x: event.changedTouches[0].clientX, y: event.changedTouches[0].clientY };
    }, { passive: true });
    track.addEventListener('touchend', (event) => {
      if (!touchStart) return;
      const dx = event.changedTouches[0].clientX - touchStart.x;
      const dy = event.changedTouches[0].clientY - touchStart.y;
      touchStart = undefined;
      if (Math.abs(dx) > 45 && Math.abs(dx) > Math.abs(dy) * 1.2) {
        const direction = dx < 0 ? 1 : -1;
        show(active + direction, direction);
      }
    });

    slides.forEach((slide, index) => {
      slide.classList.toggle('is-active', index === active);
      if (index !== active) slide.setAttribute('aria-hidden', 'true');
      slide.inert = index !== active;
    });
    track.classList.add('is-ready');
    setActive(0);
    function updateGeometry() {
      const width = track.clientWidth;
      gallery.style.setProperty('--gallery-video-width', `${width}px`);
      gallery.style.setProperty('--gallery-video-half-width', `${width / 2}px`);
      gallery.style.setProperty('--gallery-video-height', `${videos[0].getBoundingClientRect().height}px`);
    }
    updateGeometry();
    if ('ResizeObserver' in window) {
      new ResizeObserver(updateGeometry).observe(track);
    }
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
  setupCodeTracking();
  setupGalleries();
});
