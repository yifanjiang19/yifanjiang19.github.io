// Start every publication image and video in reading order, without waiting for scrolling.
(function () {
    var queue = [];
    var loading = false;

    function playVideo(video) {
        if (!video.isConnected || !video.hasAttribute("src")) return;
        var playback = video.play();
        if (playback) playback.catch(function () {
            // Keep a manual play option when the browser blocks autoplay.
            if (video.isConnected) video.controls = true;
        });
    }

    function loadNext() {
        if (loading) return;
        var media = queue.shift();
        if (!media) return;
        var source = media.getAttribute("data-src");
        if (!source) {
            loadNext();
            return;
        }

        loading = true;
        var isVideo = media.tagName === "VIDEO";
        var readyEvent = isVideo ? "loadeddata" : "load";
        var finished = false;
        var timer;
        function finish() {
            if (finished) return;
            finished = true;
            clearTimeout(timer);
            media.removeEventListener(readyEvent, finish);
            media.removeEventListener("error", finish);
            loading = false;
            loadNext();
        }

        media.addEventListener(readyEvent, finish);
        media.addEventListener("error", finish);
        // A slow or stalled resource must not block the rest of the page.
        timer = setTimeout(finish, 5000);
        media.removeAttribute("data-src");
        if (isVideo) media.preload = "auto";
        media.src = source;
        if (isVideo) {
            media.load();
            playVideo(media);
        } else if (media.complete) {
            finish();
        }
    }

    window.queuePublicationMedia = function (cards) {
        var selector = "img[data-src], video[data-src]";
        // Prioritize the current list, then finish all filtered-out publications.
        queue = Array.from(document.querySelectorAll("#main-pub-card-container " +
            "img[data-src], #main-pub-card-container video[data-src]"));
        cards.forEach(function (card) {
            card.querySelectorAll(selector).forEach(function (media) {
                if (queue.indexOf(media) === -1) queue.push(media);
            });
            card.querySelectorAll("video").forEach(function (video) {
                if (video.isConnected) playVideo(video);
                else video.pause();
            });
        });
        loadNext();
    };
}());
