(() => {
    "use strict";

    function initializeGetStarted() {
        const capabilityLinks = document.querySelectorAll(
            'a[href="#capabilities"]'
        );

        capabilityLinks.forEach((link) => {
            link.addEventListener("click", (event) => {
                const target = document.getElementById(
                    "capabilities"
                );

                if (!target) {
                    return;
                }

                event.preventDefault();

                target.scrollIntoView({
                    behavior: "smooth",
                    block: "start",
                });
            });
        });

        const capabilityCards = document.querySelectorAll(
            ".capability-card"
        );

        if (
            capabilityCards.length === 0 ||
            typeof IntersectionObserver === "undefined"
        ) {
            return;
        }

        const observer = new IntersectionObserver(
            (entries, currentObserver) => {
                entries.forEach((entry) => {
                    if (!entry.isIntersecting) {
                        return;
                    }

                    entry.target.classList.add(
                        "is-visible"
                    );

                    currentObserver.unobserve(
                        entry.target
                    );
                });
            },
            {
                threshold: 0.12,
            }
        );

        capabilityCards.forEach((card) => {
            observer.observe(card);
        });
    }

    if (
        document.readyState === "loading"
    ) {
        document.addEventListener(
            "DOMContentLoaded",
            initializeGetStarted,
            {
                once: true,
            }
        );
    } else {
        initializeGetStarted();
    }
})();