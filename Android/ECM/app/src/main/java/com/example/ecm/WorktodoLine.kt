package com.example.ecm

/**
 * Parsed worktodo line from shell-residue format.
 *
 * Example input line:
 *   echo '(2^151-1)' | .\build\Debug\ecm.exe -v -d 1 --curve mont --target gpu -sigma 3:2026 -gpucurves 1 1e4 0
 *   # 391612124215324515959
 *
 * Parsing rules:
 *   1. Strip "echo 'N' | anything.exe" prefix (path-agnostic).
 *   2. Remaining tokens are flags and positional B1 B2.
 *   3. Known flags: -v, --curve, --target, -d N, -sigma N:M, -gpucurves N,
 *      --mul X, --sqr X, --add X, --sub X, --special-mult X.
 *   4. Last 2 numeric tokens → B1, B2.
 *   5. Next line starting with '#' → expectedFactor.
 */
data class WorktodoLine(
    val nExpr: String,
    val b1: Double,
    val b2: Double,
    val gpuCurves: Int,
    val deviceIndex: Int,
    val verbose: Boolean,
    val sigma: String,          // "3:2026" or empty
    val mulPath: String,
    val sqrPath: String,
    val addPath: String,
    val subPath: String,
    val specialMultPath: String,
    val expectedFactor: String, // empty if no expectation
) {
    companion object {
        /** Regex: "echo 'N' | ...exe" — adapts to any exe path. */
        private val PREFIX_RE =
            Regex("""^\s*echo\s+'([^']+)'\s*\|\s*\S*[Ee][Cc][Mm](?:\.exe)?\s*""")

        private val FLAG_DEVICE     = Regex("""-d\s+(\d+)""")
        private val FLAG_SIGMA      = Regex("""-sigma\s+(\d+:\d+)""")
        private val FLAG_GPUCURVES  = Regex("""-gpucurves\s+(\d+)""")
        private val FLAG_MUL        = Regex("""--mul\s+(\S+)""")
        private val FLAG_SQR        = Regex("""--sqr\s+(\S+)""")
        private val FLAG_ADD        = Regex("""--add\s+(\S+)""")
        private val FLAG_SUB        = Regex("""--sub\s+(\S+)""")
        private val FLAG_SPECIAL    = Regex("""--special-mult\s+(\S+)""")
        private val NUMERIC         = Regex("""^[\d.eE+\-]+${'$'}""")

        private fun extract(re: Regex, s: String, group: Int = 1): String =
            re.find(s)?.groups?.get(group)?.value.orEmpty()

        fun parse(line: String, nextLine: String?): WorktodoLine? {
            val prefixMatch = PREFIX_RE.find(line) ?: return null
            val nExpr = prefixMatch.groupValues[1]
            if (nExpr.isEmpty()) return null

            // Strip prefix, keep flags + positional args
            val tail = line.substring(prefixMatch.range.last + 1).trim()
            val tokens = tail.split(Regex("""\s+"""))
            if (tokens.size < 2) return null

            // Last 2 tokens are B1, B2 (may be exponential notation)
            val b2 = tokens.last().toDoubleOrNull() ?: return null
            val b1 = tokens[tokens.lastIndex - 1].toDoubleOrNull() ?: return null

            // Parse flags from the full tail string
            val deviceIndex = extract(FLAG_DEVICE, tail).toIntOrNull() ?: 0
            val gpuCurves   = extract(FLAG_GPUCURVES, tail).toIntOrNull() ?: 64
            val sigma       = extract(FLAG_SIGMA, tail)
            val mulPath     = extract(FLAG_MUL, tail)
            val sqrPath     = extract(FLAG_SQR, tail)
            val addPath     = extract(FLAG_ADD, tail)
            val subPath     = extract(FLAG_SUB, tail)
            val specialMult = extract(FLAG_SPECIAL, tail)
            val verbose     = "-v" in tokens

            // Expected factor from comment line
            val expected = if (nextLine != null && nextLine.trimStart().startsWith("#")) {
                nextLine.trimStart().removePrefix("#").trim()
            } else ""

            return WorktodoLine(
                nExpr = nExpr,
                b1 = b1,
                b2 = b2,
                gpuCurves = gpuCurves,
                deviceIndex = deviceIndex,
                verbose = verbose,
                sigma = sigma,
                mulPath = mulPath,
                sqrPath = sqrPath,
                addPath = addPath,
                subPath = subPath,
                specialMultPath = specialMult,
                expectedFactor = expected,
            )
        }
    }
}
