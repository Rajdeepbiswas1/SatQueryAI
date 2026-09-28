(() => {
    "use strict";

    const API_BASE = "http://127.0.0.1:8000";
    const DASHBOARD_PAGE = "dashboard.html";

    function initializeLogin() {
        const form =
            document.getElementById("login-form");

        const username =
            document.getElementById("username");

        const password =
            document.getElementById("password");

        const status =
            document.getElementById("form-status");

        const button =
            document.getElementById("login-button");

        const buttonText =
            document.getElementById(
                "login-button-text"
            );

        const toggle =
            document.getElementById(
                "toggle-password"
            );

        if (
            !form ||
            !username ||
            !password ||
            !status ||
            !button
        ) {
            return;
        }

        async function checkExistingSession() {
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

                return response.ok;
            } catch {
                return false;
            }
        }

        checkExistingSession().then(
            (authenticated) => {
                if (authenticated) {
                    window.location.replace(
                        DASHBOARD_PAGE
                    );
                }
            }
        );

        if (toggle) {
            toggle.addEventListener(
                "click",
                () => {
                    const showing =
                        password.type === "text";

                    password.type =
                        showing
                            ? "password"
                            : "text";

                    toggle.textContent =
                        showing
                            ? "Show"
                            : "Hide";

                    toggle.setAttribute(
                        "aria-label",
                        showing
                            ? "Show password"
                            : "Hide password"
                    );
                }
            );
        }

        form.addEventListener(
            "submit",
            async (event) => {
                event.preventDefault();

                status.textContent = "";

                const usernameValue =
                    username.value.trim();

                const passwordValue =
                    password.value;

                if (
                    !usernameValue ||
                    !passwordValue
                ) {
                    status.textContent =
                        "Enter both your username and password.";

                    return;
                }

                button.disabled = true;

                if (buttonText) {
                    buttonText.textContent =
                        "Authenticating...";
                }

                try {
                    const response =
                        await fetch(
                            `${API_BASE}/auth/login`,
                            {
                                method: "POST",
                                credentials: "include",
                                headers: {
                                    "Content-Type":
                                        "application/json",
                                },
                                body: JSON.stringify(
                                    {
                                        username:
                                            usernameValue,
                                        password:
                                            passwordValue,
                                    }
                                ),
                            }
                        );

                    let data = null;

                    try {
                        data =
                            await response.json();
                    } catch {
                        data = null;
                    }

                    if (!response.ok) {
                        throw new Error(
                            data?.detail ||
                            data?.message ||
                            "Authentication failed. Please check your credentials."
                        );
                    }

                    if (buttonText) {
                        buttonText.textContent =
                            "Access granted";
                    }

                    await new Promise(
                        (resolve) => {
                            window.setTimeout(
                                resolve,
                                300
                            );
                        }
                    );

                    window.location.replace(
                        DASHBOARD_PAGE
                    );
                } catch (error) {
                    status.textContent =
                        error instanceof Error
                            ? error.message
                            : "Authentication failed.";

                    button.disabled = false;

                    if (buttonText) {
                        buttonText.textContent =
                            "Sign in";
                    }
                }
            }
        );
    }

    if (
        document.readyState === "loading"
    ) {
        document.addEventListener(
            "DOMContentLoaded",
            initializeLogin,
            {
                once: true,
            }
        );
    } else {
        initializeLogin();
    }
})();