(function () {
  "use strict";

  const state = {
    items: [],
    threshold: null,
    force: false,
    currentId: null,
    sessionTotal: null, // queue size the first time this page loaded
  };

  const el = {
    ctx: document.getElementById("ctx"),
    progress: document.getElementById("progress"),
    rail: document.getElementById("rail"),
    imagePanel: document.getElementById("imagePanel"),
    detailPanel: document.getElementById("detailPanel"),
    emptyState: document.getElementById("emptyState"),
    layout: document.getElementById("layout"),
  };

  function currentItem() {
    return state.items.find((it) => it.question_id === state.currentId) || null;
  }

  function pendingCount() {
    return state.items.filter((it) => !it.verified_at).length;
  }

  async function loadQueue(preserveId) {
    const res = await fetch("/api/queue");
    const data = await res.json();
    state.items = data.items;
    state.threshold = data.threshold;
    state.force = data.force;
    if (state.sessionTotal === null) state.sessionTotal = data.items.length;

    if (preserveId && state.items.some((it) => it.question_id === preserveId)) {
      state.currentId = preserveId;
    } else if (state.items.length > 0) {
      const firstUnverified = state.items.find((it) => !it.verified_at);
      state.currentId = firstUnverified ? firstUnverified.question_id : state.items[0].question_id;
    } else {
      state.currentId = null;
    }

    render();
  }

  function selectNext(afterId) {
    const idx = state.items.findIndex((it) => it.question_id === afterId);
    // Prefer the next unverified item after this one; wrap around once.
    const ordered = state.items.slice(idx + 1).concat(state.items.slice(0, idx + 1));
    const next = ordered.find((it) => !it.verified_at && it.question_id !== afterId);
    state.currentId = next ? next.question_id : null;
  }

  function render() {
    const total = state.items.length;
    if (total === 0) {
      el.layout.classList.add("hidden");
      el.emptyState.classList.remove("hidden");
      el.ctx.textContent = "";
      el.progress.textContent = "";
      return;
    }
    el.layout.classList.remove("hidden");
    el.emptyState.classList.add("hidden");

    const item = currentItem();
    if (!item) {
      // everything in the current queue snapshot is verified
      el.layout.classList.add("hidden");
      el.emptyState.classList.remove("hidden");
      return;
    }

    const denom = state.force ? total : Math.max(state.sessionTotal, total);
    el.progress.textContent = `${pendingCount()} / ${denom} pending \u00b7 threshold ${state.threshold}`;
    el.ctx.textContent = `${item.document_id} \u00b7 ${item.original_filename}`;

    renderRail(item);
    renderImages(item);
    renderDetail(item);
  }

  function renderRail(current) {
    el.rail.innerHTML = "";
    for (const it of state.items) {
      const btn = document.createElement("button");
      btn.className = "rail-item" + (it.question_id === current.question_id ? " active" : "");
      if (it.anomalies.length || it.ai_concerns.length) btn.classList.add("flagged");
      const dot = document.createElement("span");
      dot.className = "dot";
      const label = document.createElement("span");
      const shortDoc = it.document_id.replace("DOC-", "");
      label.textContent = `${shortDoc} \u00b7 ${it.section_type}-${it.question_number}`;
      if (it.verified_at) {
        label.style.textDecoration = "line-through";
        label.style.opacity = "0.5";
      }
      btn.appendChild(dot);
      btn.appendChild(label);
      btn.addEventListener("click", () => {
        state.currentId = it.question_id;
        render();
      });
      el.rail.appendChild(btn);
    }
  }

  function renderImages(item) {
    el.imagePanel.innerHTML = "";
    if (!item.image_urls.length) {
      const div = document.createElement("div");
      div.className = "no-image";
      div.textContent =
        "No page image available for this question. It extracted cleanly from a text layer, so there's no scan to compare against.";
      el.imagePanel.appendChild(div);
      return;
    }
    item.pages.forEach((pg, i) => {
      const label = document.createElement("div");
      label.className = "page-label";
      label.textContent = `page ${pg}`;
      const img = document.createElement("img");
      img.src = item.image_urls[i];
      img.alt = `Scanned page ${pg}`;
      img.addEventListener("error", () => {
        img.replaceWith(makeMissingImageNote(pg));
      });
      el.imagePanel.appendChild(label);
      el.imagePanel.appendChild(img);
    });
  }

  function makeMissingImageNote(pg) {
    const div = document.createElement("div");
    div.className = "no-image";
    div.textContent = `Page ${pg}'s render couldn't be found on disk.`;
    return div;
  }

  function tag(text, clean) {
    const span = document.createElement("span");
    span.className = "tag" + (clean ? " clean" : "");
    span.textContent = text;
    return span;
  }

  function renderDetail(item) {
    el.detailPanel.innerHTML = "";

    const meta = document.createElement("div");
    meta.className = "meta-line";
    meta.textContent = `${item.section_type} \u00b7 q${item.question_number}${
      item.subject ? " \u00b7 " + item.subject : ""
    }${item.topic ? " / " + item.topic : ""}`;
    el.detailPanel.appendChild(meta);

    const qtext = document.createElement("p");
    qtext.className = "question-text";
    qtext.textContent = item.question_text || "(empty question text)";
    el.detailPanel.appendChild(qtext);

    if (item.options.length) {
      el.detailPanel.appendChild(fieldLabel("Options"));
      const ul = document.createElement("ul");
      ul.className = "options-list";
      for (const opt of item.options) {
        const li = document.createElement("li");
        const isSource =
          item.source_answer &&
          item.source_answer.trim().toLowerCase() === opt.label.trim().toLowerCase();
        if (isSource) li.classList.add("is-source");
        li.textContent = `${opt.label}) ${opt.text}`;
        ul.appendChild(li);
      }
      el.detailPanel.appendChild(ul);
    }

    if (item.matching_pairs.length) {
      el.detailPanel.appendChild(fieldLabel("Matching pairs"));
      const ul = document.createElement("ul");
      ul.className = "pairs-list";
      for (const mp of item.matching_pairs) {
        const li = document.createElement("li");
        li.textContent = `${mp.left_label}) ${mp.left_text}  \u2194  ${mp.right_label}) ${mp.right_text}`;
        ul.appendChild(li);
      }
      el.detailPanel.appendChild(ul);
    }

    el.detailPanel.appendChild(fieldLabel("Source answer"));
    const sa = document.createElement("div");
    sa.className = "source-answer";
    sa.textContent = item.source_answer || "(none parsed)";
    el.detailPanel.appendChild(sa);

    const flagWrap = document.createElement("div");
    flagWrap.className = "field";
    flagWrap.appendChild(fieldLabel("Flags"));
    const tags = document.createElement("div");
    tags.className = "tags";
    if (!item.anomalies.length && !item.ai_concerns.length) {
      tags.appendChild(tag("no flags", true));
    }
    for (const a of item.anomalies) tags.appendChild(tag("anomaly: " + a));
    for (const c of item.ai_concerns) tags.appendChild(tag("ai: " + c));
    flagWrap.appendChild(tags);
    if (item.ai_confidence !== null && item.ai_confidence !== undefined) {
      const conf = document.createElement("div");
      conf.className = "meta-line";
      conf.style.marginTop = "6px";
      conf.textContent = `ai_confidence ${item.ai_confidence.toFixed(2)}`;
      flagWrap.appendChild(conf);
    }
    if (item.ai_notes) {
      const notes = document.createElement("div");
      notes.className = "ai-notes";
      notes.textContent = item.ai_notes;
      flagWrap.appendChild(notes);
    }
    el.detailPanel.appendChild(flagWrap);

    el.detailPanel.appendChild(document.createElement("hr")).className = "rule";

    renderForm(item);
  }

  function fieldLabel(text) {
    const div = document.createElement("div");
    div.className = "field-label";
    div.textContent = text;
    return div;
  }

  function renderForm(item) {
    const form = document.createElement("form");
    form.className = "verify-form";

    const answerLabel = document.createElement("label");
    answerLabel.textContent = "Verified answer";
    answerLabel.setAttribute("for", "verifiedAnswer");
    const answerInput = document.createElement("input");
    answerInput.type = "text";
    answerInput.id = "verifiedAnswer";
    answerInput.value = item.verified_answer || item.source_answer || "";

    const noteLabel = document.createElement("label");
    noteLabel.textContent = "Note (optional)";
    noteLabel.setAttribute("for", "verificationNote");
    const noteInput = document.createElement("textarea");
    noteInput.id = "verificationNote";
    noteInput.value = item.verification_note || "";
    noteInput.placeholder = "Why you changed it, or why you're escalating it as-is.";

    const actions = document.createElement("div");
    actions.className = "verify-actions";

    const skipBtn = document.createElement("button");
    skipBtn.type = "button";
    skipBtn.className = "btn";
    skipBtn.textContent = "Skip for now";
    skipBtn.addEventListener("click", () => {
      selectNext(item.question_id);
      render();
    });

    const saveBtn = document.createElement("button");
    saveBtn.type = "submit";
    saveBtn.className = "btn primary";
    saveBtn.textContent = "Save & next";

    actions.appendChild(skipBtn);
    actions.appendChild(saveBtn);

    form.appendChild(answerLabel);
    form.appendChild(answerInput);
    form.appendChild(noteLabel);
    form.appendChild(noteInput);
    form.appendChild(actions);

    if (item.verified_at) {
      const already = document.createElement("div");
      already.className = "already-verified";
      already.textContent = `Verified ${item.verified_at}`;
      form.appendChild(already);
    }

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      saveBtn.disabled = true;
      saveBtn.textContent = "Saving\u2026";
      try {
        const res = await fetch("/api/verify", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question_id: item.question_id,
            document_id: item.document_id,
            verified_answer: answerInput.value.trim(),
            verification_note: noteInput.value.trim(),
          }),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          alert("Couldn't save: " + (err.error || res.statusText));
          return;
        }
        const justVerifiedId = item.question_id;
        await loadQueue();
        if (!state.force) {
          // item dropped out of the queue entirely; pick whatever
          // loadQueue already selected.
        } else {
          selectNext(justVerifiedId);
          render();
        }
      } finally {
        saveBtn.disabled = false;
        saveBtn.textContent = "Save & next";
      }
    });

    el.detailPanel.appendChild(form);
  }

  loadQueue();
})();
