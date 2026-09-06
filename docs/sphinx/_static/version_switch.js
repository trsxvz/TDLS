// Version switch of the sidebar: open the same page in the chosen
// version of the site, or its home page when the page is absent there.
document.addEventListener("DOMContentLoaded", function () {
    const select = document.getElementById("tdls-version-switch");
    if (!select) return;
    select.addEventListener("change", function () {
        const versionRoot = new URL(select.dataset.contentRoot, window.location.href);
        const siteRoot = new URL("../", versionRoot);
        const home = new URL(select.value + "/", siteRoot);
        const page = new URL(select.dataset.page, home);
        fetch(page.href, { method: "HEAD" })
            .then((r) => { window.location.href = r.ok ? page.href : home.href; })
            .catch(() => { window.location.href = home.href; });
    });
});
