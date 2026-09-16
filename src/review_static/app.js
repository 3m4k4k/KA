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

    if (item.duplicate_count > 0) {
      const dupBanner = document.createElement("div");
      dupBanner.className = "duplicate-banner";
      const others = item.duplicate_group
        .map((m) => `${m.document_id.replace("DOC-", "")}\u00b7${m.section_type}-${m.question_number}`)
        .join(", ");
      dupBanner.textContent = `Duplicated \u2014 appears ${item.duplicate_count}\u00d7 total (also: ${others})`;
      el.detailPanel.appendChild(dupBanner);
    }

    const qtext = document.createElement("p");
    qtext.className = "question-text";
    qtext.textContent = item.verified_question_text || item.question_text || "(empty question text)";
    if (item.verified_question_text) {
      const editedTag = document.createElement("span");
      editedTag.className = "edited-marker";
      editedTag.textContent = "edited";
      qtext.appendChild(document.createTextNode(" "));
      qtext.appendChild(editedTag);
    }
    el.detailPanel.appendChild(qtext);

    if (item.options.length) {
      el.detailPanel.appendChild(fieldLabel("Options"));
      const ul = document.createElement("ul");
      ul.className = "options-list";
      const displayOptions = item.verified_options || item.options;
      displayOptions.forEach((opt, i) => {
        const li = document.createElement("li");
        const isSource =
          item.source_answer &&
          item.source_answer.trim().toLowerCase() === opt.label.trim().toLowerCase();
        if (isSource) li.classList.add("is-source");
        li.textContent = `${opt.label}) ${opt.text}`;
        ul.appendChild(li);
      });
      el.detailPanel.appendChild(ul);
    }

    if (item.matching_pairs.length) {
      el.detailPanel.appendChild(fieldLabel("Matching pairs"));
      const ul = document.createElement("ul");
      ul.className = "pairs-list";
      const displayPairs = item.verified_matching_pairs || item.matching_pairs;
      for (const mp of displayPairs) {
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

    // -- structural correction section (independent of answer verification) --

    const structuralHeading = document.createElement("div");
    structuralHeading.className = "field-label structural-heading";
    structuralHeading.textContent = "Structural correction (optional)";
    form.appendChild(structuralHeading);

    const dupLabel = document.createElement("label");
    dupLabel.textContent = "Mark as duplicate of question_id";
    dupLabel.setAttribute("for", "duplicateOf");
    const dupInput = document.createElement("input");
    dupInput.type = "text";
    dupInput.id = "duplicateOf";
    dupInput.placeholder = "e.g. DOC-56eb58bdf252-mcq-12";
    dupInput.value = item.duplicate_of_question_id || "";
    form.appendChild(dupLabel);
    form.appendChild(dupInput);

    const editTextLabel = document.createElement("label");
    editTextLabel.textContent = "Question text (edit only if the text itself is broken)";
    editTextLabel.setAttribute("for", "verifiedQuestionText");
    const editTextInput = document.createElement("textarea");
    editTextInput.id = "verifiedQuestionText";
    editTextInput.value = item.verified_question_text || item.question_text || "";
    form.appendChild(editTextLabel);
    form.appendChild(editTextInput);

    const sourceOptions = item.verified_options || item.options;
    const optionRows = []; // { labelInput, textInput }

    const optsLabel = document.createElement("div");
    optsLabel.className = "field-label";
    optsLabel.textContent = "Options (edit text/labels, or add/remove rows if the count is wrong)";
    form.appendChild(optsLabel);

    const optsWrap = document.createElement("div");
    optsWrap.className = "option-edit-list";
    form.appendChild(optsWrap);

    function addOptionRow(label, text) {
      const row = document.createElement("div");
      row.className = "option-edit-row";

      const labelInput = document.createElement("input");
      labelInput.type = "text";
      labelInput.className = "option-edit-label-input";
      labelInput.maxLength = 3;
      labelInput.value = label;

      const textInput = document.createElement("input");
      textInput.type = "text";
      textInput.value = text;

      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "option-remove-btn";
      removeBtn.title = "Remove this option";
      removeBtn.textContent = "\u2715";
      removeBtn.addEventListener("click", () => {
        row.remove();
        const idx = optionRows.findIndex((r) => r.row === row);
        if (idx !== -1) optionRows.splice(idx, 1);
      });

      row.appendChild(labelInput);
      row.appendChild(textInput);
      row.appendChild(removeBtn);
      optsWrap.appendChild(row);
      optionRows.push({ labelInput, textInput, row });
    }

    for (const opt of sourceOptions) addOptionRow(opt.label, opt.text);

    const addOptionBtn = document.createElement("button");
    addOptionBtn.type = "button";
    addOptionBtn.className = "btn option-add-btn";
    addOptionBtn.textContent = "+ Add option";
    addOptionBtn.addEventListener("click", () => addOptionRow("", ""));
    form.appendChild(addOptionBtn);

    form.appendChild(document.createElement("hr")).className = "rule";

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
    if (item.structural_reviewed_at) {
      const already = document.createElement("div");
      already.className = "already-verified";
      already.textContent = `Structural review ${item.structural_reviewed_at}`;
      form.appendChild(already);
    }

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      saveBtn.disabled = true;
      saveBtn.textContent = "Saving\u2026";
      try {
        // Question text: only counts as an override if it actually
        // differs from the original segmentation output -- typing the
        // same text back in shouldn't create a spurious "edited" tag.
        const trimmedText = editTextInput.value.trim();
        const verifiedQuestionText = trimmedText !== item.question_text.trim() ? trimmedText : "";

        // Options: rows can now be added, removed, or have their
        // label/text edited -- a blank leftover row (never touched)
        // is dropped rather than sent as an empty option. "changed"
        // means either the count differs from the original or any
        // surviving row's label/text differs positionally -- either
        // case means the reviewer actually touched the option set, so
        // send the full replacement list; otherwise send null so the
        // record keeps reflecting "no override".
        let verifiedOptions = null;
        const currentOptions = optionRows
          .map((r) => ({
            label: r.labelInput.value.trim(),
            text: r.textInput.value.trim(),
          }))
          .filter((o) => o.label !== "" || o.text !== "");
        const optionsChanged =
          currentOptions.length !== sourceOptions.length ||
          currentOptions.some(
            (o, i) => o.label !== sourceOptions[i].label || o.text !== sourceOptions[i].text
          );
        if (optionsChanged) {
          verifiedOptions = currentOptions;
        }

        const res = await fetch("/api/verify", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question_id: item.question_id,
            document_id: item.document_id,
            verified_answer: answerInput.value.trim(),
            verification_note: noteInput.value.trim(),
            duplicate_of_question_id: dupInput.value.trim(),
            verified_question_text: verifiedQuestionText,
            verified_options: verifiedOptions,
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
