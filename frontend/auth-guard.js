(() => {
    "use strict";

    const API_BASE = "http://127.0.0.1:8000";
    const LOGIN_PAGE = "login.html";
    const GET_STARTED_PAGE = "getstarted.html";

    function hideApplicationUntilAuthenticated() {
        const style =
            document.createElement("style");

        style.id =
            "satquery-auth-pending-style";

        style.textContent =
            "html.satquery-auth-pending body{" +
            "visibility:hidden!important;" +
            "}";

        document.documentElement.classList.add(
            "satquery-auth-pending"
        );

        document.head.appendChild(style);
    }

    function showApplication() {
        document.documentElement.classList.remove(
            "satquery-auth-pending"
        );

        const style =
            document.getElementById(
                "satquery-auth-pending-style"
            );

        if (style) {
            style.remove();
        }
    }

    function addLogoutButton(username) {
        const actions =
            document.querySelector(
                ".topbar-actions"
            );

        if (
            !actions ||
            document.getElementById(
                "satquery-auth-control"
            )
        ) {
            return;
        }

        const button =
            document.createElement(
                "button"
            );

        button.type = "button";

        button.id =
            "satquery-auth-control";

        button.textContent =
            "Sign out";

        button.title =
            `Sign out ${username}`;

        button.setAttribute(
            "aria-label",
            `Sign out ${username}`
        );

        button.style.cssText = `
            height: 34px;
            padding: 0 11px;
            border: 1px solid rgba(148,163,184,.14);
            border-radius: 10px;
            background: rgba(255,255,255,.018);
            color: #c4d3dd;
            font: 600 9px/1 Inter, "Segoe UI", Arial, sans-serif;
            cursor: pointer;
            transition:
                background .18s ease,
                border-color .18s ease,
                color .18s ease,
                transform .18s ease;
        `;

        button.addEventListener(
            "mouseenter",
            () => {
                button.style.color =
                    "#59c8ff";

                button.style.borderColor =
                    "rgba(89,200,255,.25)";

                button.style.background =
                    "rgba(89,200,255,.075)";

                button.style.transform =
                    "translateY(-1px)";
            }
        );

        button.addEventListener(
            "mouseleave",
            () => {
                button.style.color =
                    "#c4d3dd";

                button.style.borderColor =
                    "rgba(148,163,184,.14)";

                button.style.background =
                    "rgba(255,255,255,.018)";

                button.style.transform =
                    "translateY(0)";
            }
        );

        button.addEventListener(
            "click",
            async () => {
                button.disabled = true;

                button.textContent =
                    "Signing out...";

                try {
                    await fetch(
                        `${API_BASE}/auth/logout`,
                        {
                            method: "POST",
                            credentials: "include",
                        }
                    );
                } catch {
                    // Even if the logout request fails,
                    // leave the protected dashboard.
                } finally {
                    window.location.replace(
                        GET_STARTED_PAGE
                    );
                }
            }
        );

        actions.prepend(button);
    }

    async function checkAuthentication() {
        try {
            const response =
                await fetch(
                    `${API_BASE}/auth/me`,
                    {
                        method: "GET",
                        credentials: "include",
                        cache: "no-store",
                    }
                );

            if (!response.ok) {
                window.location.replace(
                    LOGIN_PAGE
                );

                return;
            }

            let data = null;

            try {
                data =
                    await response.json();
            } catch {
                data = null;
            }

            addLogoutButton(
                data?.username || "User"
            );

            showApplication();

        } catch {
            window.location.replace(
                LOGIN_PAGE
            );
        }
    }

    function initializeAuthenticationGuard() {
        hideApplicationUntilAuthenticated();
        checkAuthentication();
    }

    if (
        document.readyState === "loading"
    ) {
        document.addEventListener(
            "DOMContentLoaded",
            initializeAuthenticationGuard,
            {
                once: true,
            }
        );
    } else {
        initializeAuthenticationGuard();
    }
})();