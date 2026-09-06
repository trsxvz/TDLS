// Version menu of the sidebar: open the same page in the chosen version
// of the site, or its home page when the page is absent there.
document.addEventListener("DOMContentLoaded", function () {
    const menu = document.getElementById("tdls-version-switch");
    if (!menu) return;
    menu.querySelectorAll("a[data-version]").forEach(function (link) {
        link.addEventListener("click", function (event) {
            event.preventDefault();
            const versionRoot = new URL(menu.dataset.contentRoot, window.location.href);
            const siteRoot = new URL("../", versionRoot);
            const home = new URL(link.dataset.version + "/", siteRoot);
            const page = new URL(menu.dataset.page, home);
            fetch(page.href, { method: "HEAD" })
                .then((r) => { window.location.href = r.ok ? page.href : home.href; })
                .catch(() => { window.location.href = home.href; });
        });
    });
    // a click anywhere else folds the menu
    document.addEventListener("click", function (event) {
        if (!menu.contains(event.target)) menu.removeAttribute("open");
    });
});
