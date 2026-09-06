// Shared site behaviours (navbar, animations) for CUHK-X pages
(function () {
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    const navLinks = document.getElementById('navLinks');

    if (mobileMenuBtn && navLinks) {
        mobileMenuBtn.addEventListener('click', () => {
            navLinks.classList.toggle('active');
            mobileMenuBtn.textContent = navLinks.classList.contains('active') ? '\u2715' : '\u2630';
        });
        document.querySelectorAll('.nav-links a').forEach(link => {
            link.addEventListener('click', () => {
                navLinks.classList.remove('active');
                mobileMenuBtn.textContent = '\u2630';
            });
        });
    }

    const navbar = document.querySelector('.navbar');
    window.addEventListener('scroll', () => {
        if (!navbar) return;
        navbar.classList.toggle('scrolled', window.scrollY > 50);

        // scroll-spy only affects same-page anchor links
        const anchors = [...document.querySelectorAll('.nav-links a')]
            .filter(a => (a.getAttribute('href') || '').startsWith('#'));
        if (!anchors.length) return;
        let current = '';
        document.querySelectorAll('section[id]').forEach(section => {
            if (pageYOffset >= section.offsetTop - 200) current = section.id;
        });
        anchors.forEach(link => {
            link.classList.toggle('active', link.getAttribute('href') === '#' + current);
        });
    });

    // smooth scroll with navbar offset
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            const target = document.querySelector(targetId);
            if (!target) return;
            e.preventDefault();
            const top = target.getBoundingClientRect().top + window.pageYOffset
                - (navbar ? navbar.offsetHeight : 0) - 20;
            window.scrollTo({ top, behavior: 'smooth' });
        });
    });

    // reveal-on-scroll
    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) entry.target.classList.add('animate-in');
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });
    document.querySelectorAll('.animate-in').forEach(el => observer.observe(el));
})();
