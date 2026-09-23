import java.io.*;
import java.util.*;

/**
 * Expected number of guesses to solve, from a given set of still-possible answers,
 * under the repo's greedy engine policy (pick the guess that minimises the expected
 * size of the remaining answer set == minimise sum of squared pattern-bucket sizes).
 * Guesses are drawn from the candidate set (hard-mode greedy) for tractability.
 *
 * Input : a file, one game-state per line = space-separated candidate answer words.
 * Output: one double per line = expected optimal guesses remaining.
 *
 * Usage: java EGuess states.txt
 */
public class EGuess {
    static HashMap<String, Double> memo = new HashMap<>();

    public static void main(String[] args) throws Exception {
        if (args.length > 0 && args[0].equals("server")) {
            // persistent mode: read one candidate set per stdin line, reply with E per line
            BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
            BufferedWriter out = new BufferedWriter(new OutputStreamWriter(System.out));
            String line;
            while ((line = in.readLine()) != null) {
                line = line.trim();
                double e = line.isEmpty() ? 0.0 : eguess(line.split(" "));
                out.write(Double.toString(e)); out.newLine(); out.flush();
            }
            return;
        }
        BufferedReader in = new BufferedReader(new FileReader(args[0]));
        StringBuilder out = new StringBuilder();
        String line;
        while ((line = in.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) { out.append("0\n"); continue; }
            String[] set = line.split(" ");
            out.append(eguess(set)).append("\n");
        }
        in.close();
        System.out.print(out);
    }

    static double eguess(String[] set) {
        int n = set.length;
        if (n == 1) return 1.0;
        if (n == 2) return 1.5;
        String key = null;
        if (n <= 300) {  // memoise the small, frequently-recurring sets
            String[] s = set.clone();
            Arrays.sort(s);
            key = String.join(",", s);
            Double c = memo.get(key);
            if (c != null) return c;
        }
        // pick the greedy best splitter among candidate words
        String best = null;
        long bestScore = Long.MAX_VALUE;
        for (String g : set) {
            int[] bucket = new int[243];
            for (String a : set) bucket[patt(g, a)]++;
            long sc = 0;
            for (int b : bucket) sc += (long) b * b;
            if (sc < bestScore) { bestScore = sc; best = g; }
        }
        // partition by pattern under the best guess; recurse on non-winning buckets
        HashMap<Integer, ArrayList<String>> parts = new HashMap<>();
        for (String a : set) {
            int p = patt(best, a);
            parts.computeIfAbsent(p, k -> new ArrayList<>()).add(a);
        }
        double exp = 1.0;  // cost of playing `best` now
        for (Map.Entry<Integer, ArrayList<String>> e : parts.entrySet()) {
            if (e.getKey() == 242) continue;  // all-green: won on this guess
            ArrayList<String> sub = e.getValue();
            exp += (sub.size() / (double) n) * eguess(sub.toArray(new String[0]));
        }
        if (key != null) memo.put(key, exp);
        return exp;
    }

    // Wordle feedback of guess vs answer as base-3 int (green=2,yellow=1,grey=0)
    static int patt(String g, String a) {
        int[] sc = new int[5];
        boolean[] m = new boolean[5];
        for (int i = 0; i < 5; i++)
            if (g.charAt(i) == a.charAt(i)) { sc[i] = 2; m[i] = true; }
        for (int i = 0; i < 5; i++) {
            if (sc[i] == 2) continue;
            for (int j = 0; j < 5; j++)
                if (!m[j] && g.charAt(i) == a.charAt(j)) { sc[i] = 1; m[j] = true; break; }
        }
        int v = 0;
        for (int i = 0; i < 5; i++) v = v * 3 + sc[i];
        return v;
    }
}
