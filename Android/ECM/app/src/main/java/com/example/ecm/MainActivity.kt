package com.example.ecm

import android.content.Intent
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.view.inputmethod.InputMethodManager
import androidx.activity.result.contract.ActivityResultContracts
import android.widget.ArrayAdapter
import android.widget.AutoCompleteTextView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.core.view.updatePadding
import androidx.core.widget.NestedScrollView
import com.google.android.material.appbar.AppBarLayout
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.bottomnavigation.BottomNavigationView
import com.google.android.material.button.MaterialButton
import com.google.android.material.checkbox.MaterialCheckBox
import com.google.android.material.textfield.TextInputEditText
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors
import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader

class MainActivity : AppCompatActivity() {

    private lateinit var outputTextEcm: TextView
    private lateinit var outputTextBench: TextView
    private lateinit var logPathEcm: TextView
    private lateinit var logPathBench: TextView
    private lateinit var perfText: TextView
    private lateinit var perfUpdated: TextView
    private lateinit var progressEcm: ProgressBar
    private lateinit var progressBench: ProgressBar
    private lateinit var scroll: NestedScrollView
    private lateinit var inputBits: TextInputEditText
    private lateinit var inputKernelIters: TextInputEditText
    private lateinit var inputInstances: TextInputEditText
    private lateinit var inputLaunchRepeats: TextInputEditText
    private lateinit var inputNExpr: TextInputEditText
    private lateinit var inputNPreset: AutoCompleteTextView
    private lateinit var inputB1: TextInputEditText
    private lateinit var inputB2: TextInputEditText
    private lateinit var inputGpuCurves: TextInputEditText
    private lateinit var inputDeviceIndex: TextInputEditText
    private lateinit var inputGpuCkpt: TextInputEditText
    private lateinit var inputSaveFile: TextInputEditText
    private lateinit var chkSaveAppend: MaterialCheckBox
    private lateinit var inputSigma: TextInputEditText
    private lateinit var inputMulPath: AutoCompleteTextView
    private lateinit var inputSqrPath: AutoCompleteTextView
    private lateinit var inputAddPath: AutoCompleteTextView
    private lateinit var inputSubPath: AutoCompleteTextView
    private lateinit var inputSpecialMultPath: AutoCompleteTextView
    private lateinit var chkVerbose: MaterialCheckBox
    private lateinit var ecmAdvancedPanel: LinearLayout
    private lateinit var panelEcm: View
    private lateinit var panelBench: View
    private lateinit var toolbar: MaterialToolbar
    private lateinit var logStore: RunLogStore
    private val benchExecutor = Executors.newSingleThreadExecutor()
    private val perfExecutor = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())
    private val timeFormat = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
    @Volatile
    private var perfSampleRunning = false

    private val perfRefreshRunnable = object : Runnable {
        override fun run() {
            refreshPerfStats()
            mainHandler.postDelayed(this, PERF_REFRESH_MS)
        }
    }

    private val settingsLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult(),
    ) { result ->
        refreshLogUi()
        if (result.resultCode == RESULT_OK) {
            recreate()
        }
    }

    private val worktodoFilePicker = registerForActivityResult(
        ActivityResultContracts.OpenDocument(),
    ) { uri: Uri? ->
        if (uri == null) return@registerForActivityResult
        runWorktodo(uri)
    }

    /**
     * Copy worktodo templates from APK assets to externalFilesDir on first launch.
     * Users can then modify these files via file manager or adb pull/push.
     */
    private fun ensureWorktodoTemplates() {
        val dir = getExternalFilesDir(null) ?: return
        val templates = listOf("worktodo_selftest.txt", "worktodo_benchmark.txt")
        for (name in templates) {
            val dest = File(dir, name)
            if (dest.exists()) continue
            try {
                assets.open(name).use { input ->
                    dest.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
            } catch (_: Exception) {
                // Template file missing from assets — skip silently
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        setContentView(R.layout.activity_main)
        setupWindowInsets()

        toolbar = findViewById(R.id.toolbar)
        setupToolbarMenu()
        panelEcm = findViewById(R.id.panel_ecm)
        panelBench = findViewById(R.id.panel_bench)

        logStore = RunLogStore(applicationContext)
        outputTextEcm = findViewById(R.id.output_ecm)
        outputTextBench = findViewById(R.id.output_bench)
        logPathEcm = findViewById(R.id.log_path_ecm)
        logPathBench = findViewById(R.id.log_path_bench)
        refreshLogUi()
        perfText = findViewById(R.id.perf_text)
        perfUpdated = findViewById(R.id.perf_updated)
        progressEcm = findViewById(R.id.progress_ecm)
        progressBench = findViewById(R.id.progress_bench)
        scroll = findViewById(R.id.nestedScrollView)
        inputBits = findViewById(R.id.input_bits)
        inputKernelIters = findViewById(R.id.input_kernel_iters)
        inputInstances = findViewById(R.id.input_instances)
        inputLaunchRepeats = findViewById(R.id.input_launch_repeats)
        inputNExpr = findViewById(R.id.input_n_expr)
        inputNPreset = findViewById(R.id.input_n_preset)
        inputB1 = findViewById(R.id.input_b1)
        inputB2 = findViewById(R.id.input_b2)
        inputGpuCurves = findViewById(R.id.input_gpu_curves)
        inputDeviceIndex = findViewById(R.id.input_device_index)
        inputGpuCkpt = findViewById(R.id.input_gpu_ckpt)
        inputSaveFile = findViewById(R.id.input_save_file)
        chkSaveAppend = findViewById(R.id.chk_save_append)
        inputSigma = findViewById(R.id.input_sigma)
        inputMulPath = findViewById(R.id.input_mul_path)
        inputSqrPath = findViewById(R.id.input_sqr_path)
        inputAddPath = findViewById(R.id.input_add_path)
        inputSubPath = findViewById(R.id.input_sub_path)
        inputSpecialMultPath = findViewById(R.id.input_special_mult_path)
        chkVerbose = findViewById(R.id.chk_verbose)
        ecmAdvancedPanel = findViewById(R.id.ecm_advanced_panel)

        nativeInitAssets(assets, AppStoragePaths.openClCacheRoot(applicationContext).absolutePath)

        setupEcmPresets()
        setupEcmPathDropdowns()
        findViewById<MaterialButton>(R.id.btn_toggle_advanced).setOnClickListener {
            val show = ecmAdvancedPanel.visibility != View.VISIBLE
            ecmAdvancedPanel.visibility = if (show) View.VISIBLE else View.GONE
        }
        findViewById<MaterialButton>(R.id.btn_worktodo).setOnClickListener {
            worktodoFilePicker.launch(arrayOf("*/*"))
        }
        findViewById<MaterialButton>(R.id.btn_run_ecm).setOnClickListener {
            runEcm()
        }

        findViewById<MaterialButton>(R.id.btn_probe).setOnClickListener {
            runNative(Tab.ECM, sessionHeader = "OpenCL probe") { nativeProbe(openClLoadError) }
        }
        findViewById<MaterialButton>(R.id.btn_short).setOnClickListener {
            runNative(Tab.ECM, sessionHeader = "OpenCL short test") { nativeShortTest() }
        }
        findViewById<MaterialButton>(R.id.btn_addsub_32).setOnClickListener {
            runAddSubBench(limbBits = 32)
        }
        findViewById<MaterialButton>(R.id.btn_addsub_24).setOnClickListener {
            runAddSubBench(limbBits = 24)
        }
        findViewById<MaterialButton>(R.id.btn_mont_32).setOnClickListener {
            runMontSqrBench(limbBits = 32)
        }
        findViewById<MaterialButton>(R.id.btn_mont_24).setOnClickListener {
            runMontSqrBench(limbBits = 24)
        }
        findViewById<MaterialButton>(R.id.btn_bench_16).setOnClickListener {
            runBench(16)
        }
        findViewById<MaterialButton>(R.id.btn_bench_24).setOnClickListener {
            runBench(24)
        }
        findViewById<MaterialButton>(R.id.btn_bench_32).setOnClickListener {
            runBench(32)
        }

        setupBottomNav()
        showTab(Tab.ECM)

        ensureWorktodoTemplates()
        scroll.post { preventInitialKeyboard() }
        runNative(Tab.ECM, sessionHeader = "OpenCL probe", showSavedToast = false) {
            nativeProbe(openClLoadError)
        }
    }

    private fun setupWindowInsets() {
        window.statusBarColor = Color.TRANSPARENT
        window.navigationBarColor = Color.TRANSPARENT
        WindowInsetsControllerCompat(window, window.decorView).apply {
            isAppearanceLightStatusBars = false
            isAppearanceLightNavigationBars = true
        }

        val appBar = findViewById<AppBarLayout>(R.id.app_bar)
        val bottomNav = findViewById<BottomNavigationView>(R.id.bottom_nav)

        ViewCompat.setOnApplyWindowInsetsListener(appBar) { view, windowInsets ->
            val safeTop = windowInsets.getInsets(
                WindowInsetsCompat.Type.statusBars() or WindowInsetsCompat.Type.displayCutout(),
            ).top
            view.updatePadding(top = safeTop)
            windowInsets
        }

        ViewCompat.setOnApplyWindowInsetsListener(bottomNav) { view, windowInsets ->
            val safeBottom = windowInsets.getInsets(WindowInsetsCompat.Type.navigationBars()).bottom
            view.updatePadding(bottom = safeBottom)
            windowInsets
        }
    }

    private fun preventInitialKeyboard() {
        scroll.requestFocus()
        inputNExpr.clearFocus()
        inputNPreset.clearFocus()
        val imm = getSystemService(InputMethodManager::class.java)
        imm?.hideSoftInputFromWindow(scroll.windowToken, 0)
    }

    private fun outputFor(tab: Tab): TextView {
        return when (tab) {
            Tab.ECM -> outputTextEcm
            Tab.BENCH -> outputTextBench
        }
    }

    private fun channelFor(tab: Tab): RunLogStore.Channel {
        return when (tab) {
            Tab.ECM -> RunLogStore.Channel.ECM
            Tab.BENCH -> RunLogStore.Channel.BENCH
        }
    }

    private fun setupToolbarMenu() {
        toolbar.setOnMenuItemClickListener { item ->
            when (item.itemId) {
                R.id.action_settings -> {
                    settingsLauncher.launch(Intent(this, SettingsActivity::class.java))
                    true
                }
                R.id.action_about -> {
                    startActivity(Intent(this, AboutActivity::class.java))
                    true
                }
                else -> false
            }
        }
    }

    private fun refreshLogUi() {
        if (AppSettings.isLogToFileEnabled(this)) {
            logPathEcm.text = getString(
                R.string.log_path_label,
                logStore.displayPath(RunLogStore.Channel.ECM),
            )
            logPathBench.text = getString(
                R.string.log_path_label,
                logStore.displayPath(RunLogStore.Channel.BENCH),
            )
        } else {
            val disabled = getString(R.string.log_path_disabled)
            logPathEcm.text = disabled
            logPathBench.text = disabled
        }
    }

    private fun beginLoggedSession(tab: Tab, header: String? = null) {
        if (!AppSettings.isLogToFileEnabled(this)) {
            return
        }
        logStore.beginSession(channelFor(tab), header)
    }

    private fun appendToLog(tab: Tab, text: String) {
        if (text.isEmpty() || !AppSettings.isLogToFileEnabled(this)) {
            return
        }
        logStore.append(channelFor(tab), text)
    }

    private fun notifyLogSaved(tab: Tab) {
        if (!AppSettings.isLogSavedToastEnabled(this)) {
            return
        }
        val path = if (AppSettings.isLogToFileEnabled(this)) {
            logStore.displayPath(channelFor(tab))
        } else {
            getString(R.string.log_path_disabled)
        }
        Toast.makeText(this, getString(R.string.log_saved_toast, path), Toast.LENGTH_SHORT).show()
    }

    private fun scrollToOutput(@Suppress("UNUSED_PARAMETER") tab: Tab) {
        scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }
    }

    private enum class Tab {
        ECM,
        BENCH,
    }

    private fun setupBottomNav() {
        val bottomNav = findViewById<BottomNavigationView>(R.id.bottom_nav)
        bottomNav.setOnItemSelectedListener { item ->
            when (item.itemId) {
                R.id.nav_ecm -> {
                    showTab(Tab.ECM)
                    true
                }
                R.id.nav_bench -> {
                    showTab(Tab.BENCH)
                    true
                }
                else -> false
            }
        }
    }

    private fun showTab(tab: Tab) {
        when (tab) {
            Tab.ECM -> {
                panelEcm.visibility = View.VISIBLE
                panelBench.visibility = View.GONE
                toolbar.subtitle = getString(R.string.subtitle_ecm)
            }
            Tab.BENCH -> {
                panelEcm.visibility = View.GONE
                panelBench.visibility = View.VISIBLE
                toolbar.subtitle = getString(R.string.subtitle_bench)
            }
        }
        scroll.post { scroll.scrollTo(0, 0) }
    }

    override fun onResume() {
        super.onResume()
        mainHandler.removeCallbacks(perfRefreshRunnable)
        refreshPerfStats()
        mainHandler.postDelayed(perfRefreshRunnable, PERF_REFRESH_MS)
    }

    override fun onPause() {
        mainHandler.removeCallbacks(perfRefreshRunnable)
        super.onPause()
    }

    private fun refreshPerfStats() {
        if (perfSampleRunning) {
            return
        }
        perfSampleRunning = true
        perfExecutor.execute {
            try {
                val snapshot = DevicePerfMonitor.sample(applicationContext)
                val body = DevicePerfMonitor.format(snapshot)
                val stamp = getString(R.string.perf_updated, timeFormat.format(Date()))
                runOnUiThread {
                    perfText.text = body
                    perfUpdated.text = stamp
                }
            } finally {
                perfSampleRunning = false
            }
        }
    }

    private fun setupEcmPathDropdowns() {
        bindJniPathDropdown(inputMulPath) { nativeListMulPaths() }
        bindJniPathDropdown(inputSqrPath) { nativeListSqrPaths() }
        bindJniPathDropdown(inputAddPath) { nativeListAddPaths() }
        bindJniPathDropdown(inputSubPath) { nativeListSubPaths() }
        bindJniPathDropdown(inputSpecialMultPath) { nativeListSpecialMultPaths() }
    }

    private fun bindJniPathDropdown(
        dropdown: AutoCompleteTextView,
        defaultIndex: Int = 0,
        listFn: () -> String,
    ) {
        val raw = listFn()
        val items = raw.split('\n').filter { it.isNotEmpty() }
        dropdown.setAdapter(
            ArrayAdapter(this, android.R.layout.simple_dropdown_item_1line, items),
        )
        dropdown.threshold = 0
        if (items.isNotEmpty()) {
            dropdown.setText(items[defaultIndex], false)
        }
        dropdown.setOnClickListener { dropdown.showDropDown() }
        dropdown.setOnFocusChangeListener { _, hasFocus ->
            if (hasFocus) {
                dropdown.showDropDown()
            }
        }
    }

    private fun selectedPathValue(dropdown: AutoCompleteTextView): String {
        val text = dropdown.text?.toString()?.trim().orEmpty()
        return if (text == "auto" || text.isEmpty()) "" else text
    }

    private fun pathArgForNative(value: String): String {
        return when (value) {
            "", "auto" -> ""
            else -> value
        }
    }

    private fun setupEcmPresets() {
        val presets = resources.getStringArray(R.array.ecm_n_presets)
        val adapter = ArrayAdapter(this, android.R.layout.simple_dropdown_item_1line, presets)
        inputNPreset.setAdapter(adapter)
        inputNPreset.setText(presets[0], false)
        inputNPreset.setOnItemClickListener { _, _, position, _ ->
            if (position < presets.size - 1) {
                inputNExpr.setText(presets[position])
            }
        }
    }

    private fun parseNonNegativeInt(edit: TextInputEditText, fallback: Int): Int? {
        val raw = edit.text?.toString()?.trim().orEmpty()
        if (raw.isEmpty()) return fallback
        val value = raw.toIntOrNull() ?: return null
        return if (value >= 0) value else null
    }

    private fun parseDoubleField(edit: TextInputEditText, fallback: Double, allowZero: Boolean): Double? {
        val raw = edit.text?.toString()?.trim().orEmpty()
        if (raw.isEmpty()) return fallback
        val value = raw.toDoubleOrNull() ?: return null
        return if (value > 0.0 || (allowZero && value >= 0.0)) value else null
    }

    private fun runEcm() {
        val nExpr = inputNExpr.text?.toString()?.trim().orEmpty()
        if (nExpr.isEmpty()) {
            Toast.makeText(this, R.string.toast_ecm_invalid, Toast.LENGTH_SHORT).show()
            return
        }
        val b1 = parseDoubleField(inputB1, 2000.0, allowZero = false) ?: run {
            toastInvalid()
            return
        }
        val b2 = parseDoubleField(inputB2, 0.0, allowZero = true) ?: run {
            toastInvalid()
            return
        }
        val gpuCurves = parsePositiveInt(inputGpuCurves, 64) ?: run {
            toastInvalid()
            return
        }
        val deviceIndex = parseNonNegativeInt(inputDeviceIndex, 0) ?: run {
            toastInvalid()
            return
        }
        val gpuCkpt = parseDoubleField(inputGpuCkpt, 600.0, allowZero = true) ?: run {
            toastInvalid()
            return
        }

        setBusy(true, Tab.ECM)
        outputTextEcm.text = ""
        val saveFile = inputSaveFile.text?.toString()?.trim().orEmpty()
        val sessionHeader = buildString {
            append("ECM Montgomery GPU N=$nExpr B1=$b1 B2=$b2 gpucurves=$gpuCurves device=$deviceIndex")
            if (chkVerbose.isChecked) append(" -v")
            if (saveFile.isNotEmpty()) {
                append(if (chkSaveAppend.isChecked) " -savea " else " -save ")
                append(saveFile)
            }
        }
        val logCallback = EcmLogCallback { line ->
            appendToLog(Tab.ECM, line)
            mainHandler.post {
                outputTextEcm.append(line)
                scrollToOutput(Tab.ECM)
            }
        }
        benchExecutor.execute {
            beginLoggedSession(Tab.ECM, sessionHeader)
            val tail = try {
                nativeRunEcm(
                    nExpr,
                    b1,
                    b2,
                    gpuCurves,
                    deviceIndex,
                    chkVerbose.isChecked,
                    gpuCkpt,
                    inputSigma.text?.toString()?.trim().orEmpty(),
                    pathArgForNative(selectedPathValue(inputMulPath)),
                    pathArgForNative(selectedPathValue(inputSqrPath)),
                    pathArgForNative(selectedPathValue(inputAddPath)),
                    pathArgForNative(selectedPathValue(inputSubPath)),
                    pathArgForNative(selectedPathValue(inputSpecialMultPath)),
                    saveFile,
                    chkSaveAppend.isChecked,
                    logCallback,
                )
            } catch (e: Exception) {
                "Error: ${e.message}\n"
            }
            runOnUiThread {
                if (tail.isNotEmpty()) {
                    outputTextEcm.append(tail)
                    appendToLog(Tab.ECM, tail)
                }
                setBusy(false, Tab.ECM)
                scrollToOutput(Tab.ECM)
                if (AppSettings.isLogSavedToastEnabled(this)) {
                    notifyLogSaved(Tab.ECM)
                }
            }
        }
    }

    /** Regex keywords for mode detection in first non-blank line. */
    private val SELFTEST_KEYWORD  = Regex("""selftest""", RegexOption.IGNORE_CASE)
    private val BENCHMARK_KEYWORD = Regex("""benchmark""", RegexOption.IGNORE_CASE)

    /**
     * Detect worktodo mode from the FIRST non-blank line only.
     * Returns: (channel, isBenchmark, isRaw)
     *   - Comment line with "selftest" → WORKTODO_SELFTEST, isBenchmark=false, isRaw=false
     *   - Comment line with "benchmark" → WORKTODO_BENCHMARK, isBenchmark=true, isRaw=false
     *   - Comment line with no keyword → ECM (raw mode, full output)
     *   - Command line (starts with echo) → ECM (raw mode, full output)
     */
    private fun detectWorktodoMode(lines: List<String>): Triple<RunLogStore.Channel, Boolean, Boolean> {
        val first = lines.firstOrNull { it.isNotBlank() }?.trim() ?: ""
        if (first.startsWith("#")) {
            if (SELFTEST_KEYWORD.containsMatchIn(first)) {
                return Triple(RunLogStore.Channel.WORKTODO_SELFTEST, false, false)
            }
            if (BENCHMARK_KEYWORD.containsMatchIn(first)) {
                return Triple(RunLogStore.Channel.WORKTODO_BENCHMARK, true, false)
            }
        }
        // No keyword or first line is a command → raw ECM mode
        return Triple(RunLogStore.Channel.ECM, false, true)
    }

    /** Extract factor[N]=value from native output. */
    private fun extractFactor(output: String): String? {
        val re = Regex("""factor\[\d+\]\s*=\s*(\d+)""")
        return re.find(output)?.groupValues?.get(1)
    }

    /** Extract gputime=xxx ms from native output. */
    private fun extractGputime(output: String): String? {
        val re = Regex("""gputime=([\d.]+)\s*ms""")
        return re.find(output)?.groupValues?.get(1)
    }

    /**
     * Parse worktodo file from URI and execute each line sequentially.
     * Mode is detected from the FIRST non-blank line:
     *   "# selftest"   → selftest  (factor comparison, PASS/FAIL per line)
     *   "# benchmark"  → benchmark (gputime only, no factor output)
     *   anything else  → raw       (full output dumped to scroll window, like regular ECM)
     * All output is written to the log system in real-time.
     */
    private fun runWorktodo(uri: Uri) {
        outputTextEcm.text = ""
        setBusy(true, Tab.ECM)
        benchExecutor.execute {
            try {
                val lines = contentResolver.openInputStream(uri)?.use { stream ->
                    BufferedReader(InputStreamReader(stream)).readLines()
                } ?: emptyList()
                if (lines.isEmpty()) {
                    runOnUiThread {
                        outputTextEcm.text = "Error: empty file"
                        setBusy(false, Tab.ECM)
                    }
                    return@execute
                }

                val (channel, isBenchmark, isRaw) = detectWorktodoMode(lines)
                val modeName = when {
                    isRaw -> "raw"
                    isBenchmark -> "benchmark"
                    else -> "selftest"
                }
                val header = "=== Worktodo: ${uri.lastPathSegment ?: "unknown"} ($modeName) ==="

                // Start log session
                if (AppSettings.isLogToFileEnabled(this)) {
                    logStore.beginSession(channel, header)
                }

                val sb = StringBuilder()
                sb.appendLine(header)
                var pass = 0
                var fail = 0
                var total = 0

                var i = 0
                while (i < lines.size) {
                    val cur = lines[i].trim()
                    if (cur.isEmpty() || (cur.startsWith("#") && total == 0)) {
                        i++
                        continue
                    }
                    if (!cur.startsWith("echo")) {
                        i++
                        continue
                    }
                    val nextLine = if (i + 1 < lines.size) lines[i + 1].trim() else null
                    val wl = WorktodoLine.parse(cur, nextLine) ?: run {
                        val skipMsg = "  [SKIP] $cur\n"
                        sb.append(skipMsg)
                        logStore.append(channel, skipMsg)
                        i++
                        continue
                    }
                    total++
                    val progress = "[$total]"
                    val detail = "$progress N=${wl.nExpr} σ=${wl.sigma} curves=${wl.gpuCurves} B1=${wl.b1}\n"
                    sb.append(detail)
                    logStore.append(channel, detail)
                    runOnUiThread { outputTextEcm.text = sb.toString() }

                    val capture = StringBuilder()
                    val logCallback = EcmLogCallback { line ->
                        val t = line.trim()
                        if (isRaw) {
                            // Raw mode: dump all output to scroll window
                            capture.appendLine(t)
                        } else if (isBenchmark) {
                            if (t.contains("gputime=")) {
                                capture.append("  ").appendLine(t)
                            }
                        } else {
                            // Selftest
                            if (t.contains("factor[") || t.contains("gputime=")) {
                                capture.append("  ").appendLine(t)
                            }
                        }
                    }

                    val startTime = System.currentTimeMillis()
                    val tail = try {
                        nativeRunEcm(
                            wl.nExpr, wl.b1, wl.b2, wl.gpuCurves, wl.deviceIndex,
                            wl.verbose, 600.0, wl.sigma,
                            wl.mulPath, wl.sqrPath, wl.addPath, wl.subPath, wl.specialMultPath,
                            "", false, logCallback,
                        )
                    } catch (e: Exception) {
                        "Error: ${e.message}"
                    }
                    val elapsed = (System.currentTimeMillis() - startTime) / 1000.0

                    val resultLine: String
                    if (isRaw) {
                        resultLine = "  done (${elapsed}s)\n"
                    } else if (isBenchmark) {
                        resultLine = "  done (${elapsed}s)\n"
                    } else {
                        // Selftest: nativeRunEcm returns only "RESULT: OK" when logCallback
                        // is active (full output goes to callback). Extract factor from
                        // the captured callback lines, not from tail.
                        val factorFound = extractFactor(capture.toString())
                        if (factorFound != null && factorFound == wl.expectedFactor && wl.expectedFactor.isNotEmpty()) {
                            pass++
                            resultLine = "  PASS (${elapsed}s)\n"
                        } else if (wl.expectedFactor.isNotEmpty()) {
                            fail++
                            resultLine = "  FAIL expected=${wl.expectedFactor} got=${factorFound ?: "(none)"} (${elapsed}s)\n"
                        } else {
                            resultLine = "  done (${elapsed}s)\n"
                        }
                    }

                    val blk = capture.toString() + resultLine
                    sb.append(blk)
                    logStore.append(channel, blk)

                    runOnUiThread {
                        outputTextEcm.text = sb.toString()
                        scrollToOutput(Tab.ECM)
                    }
                    i++
                }

                val summary = "\n=== Results: $pass PASS, $fail FAIL, $total TOTAL ===\n"
                sb.append(summary)
                logStore.append(channel, summary)

                runOnUiThread {
                    outputTextEcm.text = sb.toString()
                    setBusy(false, Tab.ECM)
                    scrollToOutput(Tab.ECM)
                    if (AppSettings.isLogSavedToastEnabled(this)) {
                        val path = logStore.displayPath(channel)
                        Toast.makeText(
                            this, getString(R.string.log_saved_toast, path), Toast.LENGTH_SHORT
                        ).show()
                    }
                }
            } catch (e: Exception) {
                runOnUiThread {
                    outputTextEcm.text = "Error reading file: ${e.message}"
                    setBusy(false, Tab.ECM)
                }
            }
        }
    }

    private fun parsePositiveInt(edit: TextInputEditText, fallback: Int): Int? {
        val raw = edit.text?.toString()?.trim().orEmpty()
        if (raw.isEmpty()) return fallback
        val value = raw.toIntOrNull() ?: return null
        return if (value > 0) value else null
    }

    private fun runMontSqrBench(limbBits: Int) {
        val defaultBits = 512
        val bits = parsePositiveInt(inputBits, defaultBits) ?: run {
            toastInvalid()
            return
        }
        val kernelIters = parsePositiveInt(inputKernelIters, 1000) ?: run {
            toastInvalid()
            return
        }
        val instances = parsePositiveInt(inputInstances, 128) ?: run {
            toastInvalid()
            return
        }
        val launchRepeats = parsePositiveInt(inputLaunchRepeats, 1) ?: run {
            toastInvalid()
            return
        }
        if (limbBits == 24) {
            if (bits != 512 && bits % 24 != 0) {
                Toast.makeText(this, R.string.toast_mont_i24_bits, Toast.LENGTH_SHORT).show()
                return
            }
        } else if (bits % 32 != 0) {
            Toast.makeText(this, getString(R.string.toast_bits_multiple, 32), Toast.LENGTH_SHORT).show()
            return
        }
        val useWg = limbBits == 32 && bits != 512
        runNative(
            Tab.BENCH,
            sessionHeader = "Montgomery bench bits=$bits limbBits=$limbBits iters=$kernelIters inst=$instances repeats=$launchRepeats",
        ) {
            nativeMontSqrBench(bits, kernelIters, instances, launchRepeats, useWg, tpi = 4, limbBits)
        }
    }

    private fun runAddSubBench(limbBits: Int) {
        val defaultBits = if (limbBits == 24) 504 else 512
        val bits = parsePositiveInt(inputBits, defaultBits) ?: run {
            toastInvalid()
            return
        }
        val kernelIters = parsePositiveInt(inputKernelIters, 1000) ?: run {
            toastInvalid()
            return
        }
        val instances = parsePositiveInt(inputInstances, 128) ?: run {
            toastInvalid()
            return
        }
        val launchRepeats = parsePositiveInt(inputLaunchRepeats, 1) ?: run {
            toastInvalid()
            return
        }
        if (bits % limbBits != 0) {
            Toast.makeText(this, getString(R.string.toast_bits_multiple, limbBits), Toast.LENGTH_SHORT).show()
            return
        }
        runNative(
            Tab.BENCH,
            sessionHeader = "AddSub bench bits=$bits limbBits=$limbBits iters=$kernelIters inst=$instances repeats=$launchRepeats",
        ) {
            nativeAddSubBench(bits, kernelIters, instances, launchRepeats, limbBits)
        }
    }

    private fun toastInvalid() {
        Toast.makeText(this, R.string.invalid_params, Toast.LENGTH_SHORT).show()
    }

    private fun runBench(bits: Int) {
        runNative(Tab.BENCH, sessionHeader = "Limb add-mod bench limbBits=$bits") {
            nativeBitBench(bits, elements = 1 shl 18, kernelIters = 64, launchRepeats = 8)
        }
    }

    private fun runNative(
        tab: Tab,
        sessionHeader: String? = null,
        showSavedToast: Boolean = true,
        block: () -> String,
    ) {
        val output = outputFor(tab)
        setBusy(true, tab)
        benchExecutor.execute {
            beginLoggedSession(tab, sessionHeader)
            val result = try {
                block()
            } catch (e: Exception) {
                "Error: ${e.message}"
            }
            appendToLog(tab, result)
            if (!result.endsWith("\n")) {
                appendToLog(tab, "\n")
            }
            runOnUiThread {
                output.text = result
                setBusy(false, tab)
                scrollToOutput(tab)
                if (showSavedToast && AppSettings.isLogSavedToastEnabled(this)) {
                    notifyLogSaved(tab)
                }
            }
        }
    }

    private fun setBusy(busy: Boolean, tab: Tab) {
        when (tab) {
            Tab.ECM -> progressEcm.visibility = if (busy) View.VISIBLE else View.GONE
            Tab.BENCH -> progressBench.visibility = if (busy) View.VISIBLE else View.GONE
        }
        findViewById<MaterialButton>(R.id.btn_probe).isEnabled = !busy
        findViewById<MaterialButton>(R.id.btn_short).isEnabled = !busy
        findViewById<MaterialButton>(R.id.btn_addsub_32).isEnabled = !busy
        findViewById<MaterialButton>(R.id.btn_addsub_24).isEnabled = !busy
        findViewById<MaterialButton>(R.id.btn_mont_32).isEnabled = !busy
        findViewById<MaterialButton>(R.id.btn_mont_24).isEnabled = !busy
        findViewById<MaterialButton>(R.id.btn_bench_16).isEnabled = !busy
        findViewById<MaterialButton>(R.id.btn_bench_24).isEnabled = !busy
        findViewById<MaterialButton>(R.id.btn_bench_32).isEnabled = !busy
        findViewById<MaterialButton>(R.id.btn_run_ecm).isEnabled = !busy
        inputBits.isEnabled = !busy
        inputKernelIters.isEnabled = !busy
        inputInstances.isEnabled = !busy
        inputLaunchRepeats.isEnabled = !busy
        inputNExpr.isEnabled = !busy
        inputNPreset.isEnabled = !busy
        inputB1.isEnabled = !busy
        inputB2.isEnabled = !busy
        inputGpuCurves.isEnabled = !busy
        inputDeviceIndex.isEnabled = !busy
        inputGpuCkpt.isEnabled = !busy
        inputSigma.isEnabled = !busy
        inputMulPath.isEnabled = !busy
        inputSqrPath.isEnabled = !busy
        inputAddPath.isEnabled = !busy
        inputSubPath.isEnabled = !busy
        inputSpecialMultPath.isEnabled = !busy
        chkVerbose.isEnabled = !busy
    }

    override fun onDestroy() {
        mainHandler.removeCallbacks(perfRefreshRunnable)
        benchExecutor.shutdownNow()
        perfExecutor.shutdownNow()
        super.onDestroy()
    }

    private external fun nativeInitAssets(
        assetManager: android.content.res.AssetManager,
        cacheDir: String,
    )
    private external fun nativeProbe(openClLoadError: String?): String
    private external fun nativeShortTest(): String
    private external fun nativeListMulPaths(): String
    private external fun nativeListSqrPaths(): String
    private external fun nativeListAddPaths(): String
    private external fun nativeListSubPaths(): String
    private external fun nativeListSpecialMultPaths(): String
    private external fun nativeMontSqrBench(
        bits: Int,
        kernelIters: Int,
        instances: Int,
        launchRepeats: Int,
        useWg: Boolean,
        tpi: Int,
        limbBits: Int,
    ): String

    private external fun nativeAddSubBench(
        bits: Int,
        kernelIters: Int,
        instances: Int,
        launchRepeats: Int,
        limbBits: Int,
    ): String
    private external fun nativeBitBench(
        limbBits: Int,
        elements: Int,
        kernelIters: Int,
        launchRepeats: Int,
    ): String

    private external fun nativeRunEcm(
        nExpr: String,
        b1: Double,
        b2: Double,
        gpuCurves: Int,
        deviceIndex: Int,
        verbose: Boolean,
        gpuCkptSec: Double,
        sigma: String,
        mulPath: String,
        sqrPath: String,
        addPath: String,
        subPath: String,
        specialMultPath: String,
        saveFile: String,
        saveAppend: Boolean,
        logCallback: EcmLogCallback?,
    ): String

    companion object {
        private const val PERF_REFRESH_MS = 1500L

        @JvmField
        var openClLoadError: String? = null

        init {
            try {
                System.loadLibrary("OpenCL")
            } catch (e: UnsatisfiedLinkError) {
                openClLoadError = e.message
            }
            System.loadLibrary("ecm")
        }
    }
}
