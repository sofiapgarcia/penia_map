import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert, argrelmin
import matplotlib.pyplot as plt

EMG_COLUMNS = ['EMG_Extensor_filt', 'EMG_Flexor_filt']


# --------------------------------------------------------------------------- #
# Bloco base: TKEO, suavização e o algoritmo detect_onset_bp fornecido
# --------------------------------------------------------------------------- #

def _tkeo(x):
    """
    Aplica o Teager-Kaiser Energy Operator (TKEO) a um sinal 1D.

    :param x: array 1D com o sinal.
    :return: array 1D com a saída do TKEO (mesmo tamanho de x; a primeira e a
        última amostra ficam a 0).
    """
    tkeo = np.zeros_like(x, dtype=float)
    tkeo[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]
    return tkeo


def smooth_signal(signal, window_size=25):
    """
    Suaviza um sinal usando uma média móvel simples.

    :param signal: array com o sinal de entrada.
    :param window_size: tamanho da janela da média móvel (em amostras).
    :return: array com o sinal suavizado (mesmo tamanho de `signal`).
    """
    window_size = max(1, int(window_size))
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')


def _bp_pipeline_components(s, phase_sigma, imag_sigma, smooth_window):
    """
    Calcula os passos intermédios do algoritmo `detect_onset_bp` (sem a parte
    de deteção de mínimos), para poderem ser reutilizados tanto na deteção
    como no plot de diagnóstico do pipeline.

    :param s: array 1D com o sinal de EMG.
    :param phase_sigma: ver `detect_onset_bp`.
    :param imag_sigma: ver `detect_onset_bp`.
    :param smooth_window: ver `detect_onset_bp`.
    :return: tuplo (x_feat, x_smooth, ins_phase_s, imag_part_s):
        - x_feat: |TKEO| (envolvente "em bruto", antes da média móvel).
        - x_smooth: |TKEO| suavizado pela média móvel (`smooth_signal`).
        - ins_phase_s: fase instantânea do sinal analítico (Hilbert),
          suavizada com filtro Gaussiano.
        - imag_part_s: parte imaginária do sinal analítico, suavizada com
          filtro Gaussiano.
    """
    x_tkeo = _tkeo(s)
    x_feat = np.abs(x_tkeo)
    x_smooth = smooth_signal(x_feat, smooth_window)
    a_sig1 = hilbert(x_smooth)

    ins_phase = np.unwrap(np.angle(a_sig1))
    ins_phase_s = gaussian_filter1d(ins_phase, sigma=phase_sigma)

    imag_part = np.imag(a_sig1)
    imag_part_s = gaussian_filter1d(imag_part, sigma=imag_sigma)

    return x_feat, x_smooth, ins_phase_s, imag_part_s


def _pair_minima(mini_phase, mini_imag, max_pair_distance):
    """
    Combina os mínimos locais da fase instantânea (`mini_phase`) com os da
    parte imaginária (`mini_imag`), emparelhando cada um com o mínimo mais
    próximo NO TEMPO do outro array -- em vez de os emparelhar por posição
    (1º com 1º, 2º com 2º, ...), que é frágil: basta um mínimo espúrio a
    mais (ou a menos) nalguma repetição anterior para desalinhar todos os
    pares seguintes, produzindo onsets sem qualquer sentido temporal.

    Um par só é aceite se a distância entre os dois mínimos for, no máximo,
    `max_pair_distance` amostras; caso contrário, esse mínimo de fase fica
    sem correspondência e é descartado (não entra no resultado).

    :param mini_phase: array 1D (ordenado ou não) com os índices dos mínimos
        locais da fase instantânea suavizada.
    :param mini_imag: array 1D (ordenado ou não) com os índices dos mínimos
        locais da parte imaginária suavizada.
    :param max_pair_distance: distância máxima (em amostras) entre um mínimo
        de fase e um mínimo de parte imaginária para serem considerados o
        mesmo evento.
    :return: array 1D ordenado com o índice combinado (média) de cada par
        aceite. Vazio se não houver nenhum par dentro da distância máxima.
    """
    mini_phase = np.asarray(mini_phase)
    mini_imag = np.asarray(mini_imag)

    if mini_phase.size == 0 or mini_imag.size == 0:
        return np.array([], dtype=int)

    pairs = []
    for mp in mini_phase:
        j = int(np.argmin(np.abs(mini_imag - mp)))
        if abs(mini_imag[j] - mp) <= max_pair_distance:
            pairs.append((mp + mini_imag[j]) // 2)

    return np.sort(np.array(pairs, dtype=int))


def detect_onset_bp(s, order=900, phase_sigma=150, imag_sigma=100, smooth_window=25,
                     agonist=True, plot=False, max_pair_distance=None):
    """
    Deteta o(s) onset(s) de EMG a partir da envolvente do TKEO usando a
    transformada de Hilbert: a fase instantânea e a parte imaginária do sinal
    analítico são suavizadas com um filtro Gaussiano, e os seus mínimos locais
    (início da ativação) são usados como candidatos a onset. O onset final
    ("agonist") é a média, amostra a amostra, entre os mínimos da fase e os
    mínimos da parte imaginária.

    Esta função replica o algoritmo fornecido; o único acrescento é uma
    proteção para o caso de `mini_phase` e `mini_imag` terem números de picos
    diferentes (o código original assumia sempre o mesmo número).

    :param s: array 1D com o sinal de EMG (bruto ou já filtrado).
    :param order: número de amostras usado em `argrelmin` para considerar um
        ponto como mínimo local (equivalente ao parâmetro `order` do
        `scipy.signal.argrelmin`; tipicamente elevado para eliminar mínimos
        espúrios).
    :param phase_sigma: sigma (em amostras) do filtro Gaussiano aplicado à
        fase instantânea.
    :param imag_sigma: sigma (em amostras) do filtro Gaussiano aplicado à
        parte imaginária do sinal analítico.
    :param smooth_window: tamanho da janela (em amostras) da média móvel
        usada para suavizar a envolvente |TKEO|.
    :param agonist: se True, devolve o onset combinado (média entre os
        mínimos da fase e da parte imaginária); se False, devolve apenas os
        mínimos da fase (equivalente ao `agonist=False` do algoritmo
        original).
    :param plot: se True, mostra um gráfico de diagnóstico com o sinal, a
        fase e a parte imaginária normalizados, e os onsets marcados.
    :param max_pair_distance: distância máxima (em amostras) entre um mínimo
        da fase e um mínimo da parte imaginária para serem considerados o
        mesmo evento (e por isso combinados por média). Por omissão (None),
        usa-se o próprio `order`. Mínimos sem correspondência dentro desta
        distância são descartados, em vez de combinados às cegas por
        posição no array (que desalinharia todos os onsets seguintes assim
        que um dos dois sinais apanhasse um mínimo espúrio a mais ou a
        menos nalguma repetição anterior).
    :return: tuplo (ins_phase_s, onsets), em que `ins_phase_s` é a fase
        instantânea suavizada e `onsets` é o array (possivelmente vazio) de
        índices de onset detetados.
    """
    # Passo 1 - obter as componentes analíticas a partir da envolvente do TKEO
    x_feat, x_smooth, ins_phase_s, imag_part_s = _bp_pipeline_components(
        s, phase_sigma, imag_sigma, smooth_window
    )

    # Mínimos locais das componentes suavizadas (início da fase de ativação)
    mini_phase = argrelmin(ins_phase_s, order=order)[0]
    mini_imag = argrelmin(imag_part_s, order=order)[0]

    # Combina os dois conjuntos de mínimos, emparelhando por proximidade no
    # tempo (não por posição no array -- ver `_pair_minima`)
    pair_distance = order if max_pair_distance is None else max_pair_distance
    mini = _pair_minima(mini_phase, mini_imag, pair_distance)

    if plot:
        def _norm(v):
            m = np.max(np.abs(v))
            return v / m if m != 0 else v

        plt.figure()
        plt.plot(_norm(s), label='EMG')
        plt.plot(_norm(ins_phase_s), label='Fase instantânea (suavizada)')
        plt.plot(_norm(imag_part_s), label='Parte imaginária (suavizada)')
        if mini.size:
            plt.vlines(mini, -1, 1, colors='r', label='EMG Onset')
        plt.legend()
        plt.show()

    if not agonist:
        return ins_phase_s, mini_phase
    return ins_phase_s, mini


def detect_offset_bp(s, order=900, phase_sigma=150, imag_sigma=100, smooth_window=25,
                      max_pair_distance=None):
    """
    Deteta o(s) offset(s) de EMG aplicando `detect_onset_bp` ao sinal
    invertido no tempo: um "onset" no sinal invertido corresponde,
    fisiologicamente, ao fim da contração (queda de atividade) no sinal
    original. Não faz parte do algoritmo fornecido, mas é a extensão natural
    do mesmo, necessária para as condições isométricas (HI), onde também
    interessa saber quando termina cada contração.

    :param s: array 1D com o sinal de EMG.
    :param order: ver `detect_onset_bp`.
    :param phase_sigma: ver `detect_onset_bp`.
    :param imag_sigma: ver `detect_onset_bp`.
    :param smooth_window: ver `detect_onset_bp`.
    :param max_pair_distance: ver `detect_onset_bp`.
    :return: array (ordenado) de índices de offset, já no referencial do
        sinal original (não invertido).
    """
    s_rev = s[::-1]
    _, mini_rev = detect_onset_bp(s_rev, order=order, phase_sigma=phase_sigma,
                                   imag_sigma=imag_sigma, smooth_window=smooth_window,
                                   agonist=True, plot=False, max_pair_distance=max_pair_distance)
    n = s.size
    offsets = np.sort(n - 1 - np.asarray(mini_rev))
    return offsets


def _emg_envelope(x, fs, window_ms):
    """
    Constrói a envolvente suavizada |TKEO| de um sinal de EMG.

    Usada apenas para estimar, de forma aproximada, o número e a localização
    temporal das contrações isométricas em HI (via limiar robusto); a
    deteção fina de onset/offset é sempre feita por `detect_onset_bp` /
    `detect_offset_bp`.

    :param x: array 1D com o sinal de EMG.
    :param fs: frequência de amostragem (Hz).
    :param window_ms: janela da média móvel (ms) usada para suavizar |TKEO|.
    :return: array 1D com a envolvente suavizada (mesmo tamanho de x).
    """
    rectified = np.abs(x)
    tkeo_rectified = np.abs(_tkeo(rectified))
    window_samples = max(1, int(round((window_ms / 1000.0) * fs)))
    return smooth_signal(tkeo_rectified, window_samples)


def _robust_baseline_threshold(envelope, threshold_k):
    """
    Estima um limiar de deteção robusto para uma envolvente, com base na
    mediana e no desvio absoluto mediano (MAD).

    :param envelope: array 1D com a envolvente (suavizada).
    :param threshold_k: multiplicador aplicado à estimativa robusta de
        dispersão.
    :return: valor escalar do limiar.
    """
    baseline = np.median(envelope)
    mad = np.median(np.abs(envelope - baseline))
    robust_std = mad * 1.4826
    return baseline + threshold_k * robust_std


def _detect_active_segments(envelope, fs, threshold, min_duration_ms, min_gap_ms):
    """
    Deteta segmentos contíguos onde a envolvente ultrapassa um limiar,
    fundindo segmentos próximos e descartando segmentos demasiado curtos.

    Usada apenas para localizar aproximadamente os plateaus de ativação em
    condições isométricas (HI), de forma a saber a quantas contrações
    (repetições) associar cada onset/offset devolvido por `detect_onset_bp` /
    `detect_offset_bp`.

    :param envelope: array 1D com a envolvente (suavizada).
    :param fs: frequência de amostragem (Hz).
    :param threshold: limiar de amplitude acima do qual o sinal é
        considerado "ativo".
    :param min_duration_ms: duração mínima (ms) para um segmento ser mantido
        como contração real.
    :param min_gap_ms: intervalo máximo (ms) entre dois segmentos ativos
        para serem fundidos num só.
    :return: lista de tuplos (start_idx, end_idx) (end exclusive), ordenada
        por start_idx.
    """
    above = envelope > threshold
    diff = np.diff(above.astype(int))
    starts = list(np.flatnonzero(diff == 1) + 1)
    ends = list(np.flatnonzero(diff == -1) + 1)
    if above.size and above[0]:
        starts = [0] + starts
    if above.size and above[-1]:
        ends = ends + [above.size]

    segments = list(zip(starts, ends))
    if not segments:
        return []

    min_gap_samples = (min_gap_ms / 1000.0) * fs
    merged = [segments[0]]
    for s, e in segments[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e < min_gap_samples:
            merged[-1] = (prev_s, e)
        else:
            merged.append((s, e))

    min_duration_samples = (min_duration_ms / 1000.0) * fs
    merged = [(s, e) for s, e in merged if (e - s) >= min_duration_samples]

    return merged


def _rep_numbers_from_phase(phase_col):
    """
    Extrai a lista ordenada de números de repetição a partir da coluna
    'Phase'.

    :param phase_col: array 1D com os valores da coluna 'Phase' (ex: 'C1',
        'E1', 'R', ...).
    :return: lista ordenada de números de repetição encontrados (com base
        nos rótulos 'C{rep}').
    """
    reps = set()
    for label in np.unique(phase_col):
        if isinstance(label, str) and label.startswith('C'):
            reps.add(int(label[1:]))
    return sorted(reps)


# --------------------------------------------------------------------------- #
# Deteção por condição
# --------------------------------------------------------------------------- #

def _detect_emg_onsets_single_condition(df, fs, columns, order, phase_sigma, imag_sigma,
                                         smooth_window, max_onset_lookback_ms=300,
                                         max_pair_distance=None):
    """
    Deteta o onset de EMG (via `detect_onset_bp`) para ambos os músculos, em
    cada repetição, numa condição com coluna 'Phase' (ex: HD), onde o início
    da fase concêntrica de cada repetição é conhecido.

    `detect_onset_bp` é aplicado ao sinal COMPLETO da condição, para cada
    músculo (o algoritmo precisa de um sinal relativamente longo para que os
    filtros Gaussianos e o `argrelmin` com `order` elevado funcionem bem).
    Para cada repetição, escolhe-se o onset detetado mais próximo (e
    anterior) ao início da respetiva fase concêntrica -- mas só se esse
    onset cair dentro de uma janela máxima de `max_onset_lookback_ms`
    milissegundos antes desse início. Isto evita que um mínimo global
    "apanhado" muito longe no tempo (de outra repetição, ou de uma oscilação
    lenta da fase que não corresponde a nenhuma ativação real) seja
    incorretamente aceite como onset.

    :param df: DataFrame com a coluna 'Phase' e as colunas de EMG a
        processar.
    :param fs: frequência de amostragem (Hz).
    :param columns: nomes das colunas de EMG a processar.
    :param order: ver `detect_onset_bp`.
    :param phase_sigma: ver `detect_onset_bp`.
    :param imag_sigma: ver `detect_onset_bp`.
    :param smooth_window: ver `detect_onset_bp`.
    :param max_onset_lookback_ms: janela máxima (ms), antes do início da
        fase concêntrica, dentro da qual um onset detetado ainda é aceite
        como pertencendo a essa repetição. Se não existir nenhum onset
        dentro dessa janela, `onset_idx` fica a None para essa repetição
        (em vez de "agarrar" um onset distante e provavelmente errado).
    :param max_pair_distance: ver `detect_onset_bp`.
    :return: lista de dicts, um por (repetição, músculo), com as chaves:
        'rep', 'muscle', 'onset_idx' (índice global, ou None se não
        encontrado dentro da janela), 'concentric_start_idx',
        'concentric_end_idx', 'onset_to_start_ms'.
    """
    if 'Phase' not in df.columns:
        raise ValueError("df has no 'Phase' column -- run detect_concentric_eccentric_phases first.")
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in the data.")

    lookback_samples = int(round((max_onset_lookback_ms / 1000.0) * fs))

    phase_col = df['Phase'].to_numpy()
    reps = _rep_numbers_from_phase(phase_col)
    if not reps:
        raise ValueError("No repetitions ('C1', 'C2', ...) found in the 'Phase' column.")

    onsets_by_col = {}
    for col in columns:
        signal = df[col].to_numpy()
        _, mini = detect_onset_bp(signal, order=order, phase_sigma=phase_sigma,
                                   imag_sigma=imag_sigma, smooth_window=smooth_window,
                                   agonist=True, plot=False, max_pair_distance=max_pair_distance)
        onsets_by_col[col] = np.sort(np.asarray(mini))

    results = []
    for rep_num in reps:
        rep_idx = np.flatnonzero(phase_col == f'C{rep_num}')
        concentric_start = int(rep_idx[0])
        concentric_end = int(rep_idx[-1])

        for col in columns:
            onsets = onsets_by_col[col]
            candidates = onsets[(onsets <= concentric_start)
                                 & (onsets >= concentric_start - lookback_samples)]
            if candidates.size:
                onset_idx = int(candidates[-1])
                onset_to_start_ms = (concentric_start - onset_idx) / fs * 1000.0
            else:
                onset_idx = None
                onset_to_start_ms = None
                print(f"Warning: no onset found for '{col}' rep {rep_num} within "
                      f"{max_onset_lookback_ms} ms before the concentric start "
                      f"(idx {concentric_start}).")

            results.append({
                'rep': rep_num,
                'muscle': col,
                'onset_idx': onset_idx,
                'concentric_start_idx': concentric_start,
                'concentric_end_idx': concentric_end,
                'onset_to_start_ms': onset_to_start_ms,
            })

    return results


def _detect_emg_onsets_isometric_condition(df, fs, columns, order, phase_sigma, imag_sigma,
                                            smooth_window, window_ms, n_reps, threshold_k,
                                            min_duration_ms, min_gap_ms,
                                            max_onset_lookback_ms=300,
                                            max_offset_lookahead_ms=300,
                                            max_pair_distance=None):
    """
    Deteta onset E offset de EMG (via `detect_onset_bp` / `detect_offset_bp`)
    para condições isométricas (ex: HI), onde não existe coluna 'Phase'.

    Como `detect_onset_bp`/`detect_offset_bp` devolvem todos os onsets/offsets
    encontrados no sinal completo (sem saber a que repetição pertence cada
    um), o número aproximado e a localização temporal de cada contração são
    primeiro estimados com um limiar robusto sobre a envolvente |TKEO| (mesma
    lógica do ficheiro original). Cada plateau é depois associado ao
    onset/offset global mais próximo -- mas só se esse onset/offset cair
    dentro de uma janela máxima (`max_onset_lookback_ms` antes do início do
    plateau; `max_offset_lookahead_ms` depois do fim do plateau). Isto evita
    aceitar um onset/offset distante, "apanhado" de outra contração ou de uma
    oscilação da fase sem correspondência real.

    :param df: DataFrame com as colunas de EMG a processar.
    :param fs: frequência de amostragem (Hz).
    :param columns: nomes das colunas de EMG a processar.
    :param order: ver `detect_onset_bp`.
    :param phase_sigma: ver `detect_onset_bp`.
    :param imag_sigma: ver `detect_onset_bp`.
    :param smooth_window: ver `detect_onset_bp`.
    :param window_ms: janela (ms) usada para construir a envolvente auxiliar
        que serve apenas para localizar os plateaus (contrações).
    :param n_reps: número esperado de contrações isométricas por músculo.
    :param threshold_k: multiplicador do desvio robusto (MAD) para o limiar
        de deteção de atividade.
    :param min_duration_ms: duração mínima (ms) de um segmento ativo para
        ser aceite como contração real.
    :param min_gap_ms: intervalo máximo (ms) entre segmentos ativos para
        serem fundidos numa só contração.
    :param max_onset_lookback_ms: janela máxima (ms), antes do início do
        plateau, dentro da qual um onset ainda é aceite. Se nenhum onset cair
        dentro dela, `onset_idx` fica a None para essa repetição.
    :param max_offset_lookahead_ms: janela máxima (ms), depois do fim do
        plateau, dentro da qual um offset ainda é aceite. Se nenhum offset
        cair dentro dela, `offset_idx` fica a None para essa repetição.
    :param max_pair_distance: ver `detect_onset_bp`.
    :return: lista de dicts, um por (repetição, músculo), com as chaves:
        'rep', 'muscle', 'onset_idx' (ou None), 'offset_idx' (ou None),
        'concentric_start_idx' (início do plateau, mantido por
        compatibilidade -- NÃO é uma verdadeira fase concêntrica),
        'onset_to_start_ms'.
    """
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in the data.")

    onset_lookback_samples = int(round((max_onset_lookback_ms / 1000.0) * fs))
    offset_lookahead_samples = int(round((max_offset_lookahead_ms / 1000.0) * fs))

    results = []
    for col in columns:
        signal = df[col].to_numpy()

        # 1) localizar aproximadamente as contrações (plateaus), só para
        #    saber quantas há e onde procurar o onset/offset mais próximo.
        envelope = _emg_envelope(signal, fs, window_ms)
        threshold = _robust_baseline_threshold(envelope, threshold_k)
        segments = _detect_active_segments(envelope, fs, threshold, min_duration_ms, min_gap_ms)

        if len(segments) != n_reps:
            print(f"Warning: expected {n_reps} contractions for '{col}', found {len(segments)}. "
                  f"Adjust threshold_k / min_duration_ms / min_gap_ms if this looks wrong. "
                  f"Keeping the {n_reps} longest segments found, in temporal order.")
            segments = sorted(
                sorted(segments, key=lambda se: se[1] - se[0], reverse=True)[:n_reps],
                key=lambda se: se[0],
            )

        # 2) onset/offset "verdadeiros", via detect_onset_bp / detect_offset_bp,
        #    aplicados ao sinal completo do músculo.
        _, onsets = detect_onset_bp(signal, order=order, phase_sigma=phase_sigma,
                                     imag_sigma=imag_sigma, smooth_window=smooth_window,
                                     agonist=True, plot=False, max_pair_distance=max_pair_distance)
        onsets = np.sort(np.asarray(onsets))
        offsets = detect_offset_bp(signal, order=order, phase_sigma=phase_sigma,
                                    imag_sigma=imag_sigma, smooth_window=smooth_window,
                                    max_pair_distance=max_pair_distance)

        for rep_num, (seg_start, seg_end) in enumerate(segments, start=1):
            onset_candidates = onsets[(onsets >= seg_start - onset_lookback_samples)
                                       & (onsets <= seg_end)]
            if onset_candidates.size:
                onset_idx = int(onset_candidates[np.argmin(np.abs(onset_candidates - seg_start))])
            else:
                onset_idx = None
                print(f"Warning: no onset found for '{col}' rep {rep_num} within "
                      f"{max_onset_lookback_ms} ms before the plateau start (idx {seg_start}).")

            offset_candidates = offsets[(offsets >= seg_start)
                                         & (offsets <= seg_end + offset_lookahead_samples)]
            if offset_candidates.size:
                offset_idx = int(offset_candidates[np.argmin(np.abs(offset_candidates - seg_end))])
            else:
                offset_idx = None
                print(f"Warning: no offset found for '{col}' rep {rep_num} within "
                      f"{max_offset_lookahead_ms} ms after the plateau end (idx {seg_end}).")

            onset_to_start_ms = ((seg_start - onset_idx) / fs * 1000.0
                                  if onset_idx is not None else None)

            results.append({
                'rep': rep_num,
                'muscle': col,
                'onset_idx': onset_idx,
                'offset_idx': offset_idx,
                'concentric_start_idx': seg_start,
                'onset_to_start_ms': onset_to_start_ms,
            })

    return results


# --------------------------------------------------------------------------- #
# Função principal
# --------------------------------------------------------------------------- #

def detect_emg_onsets(subject_data, conditions=('HD', 'HI'), order=900, phase_sigma=150,
                       imag_sigma=100, smooth_window=25, emg_columns=None, show_plot=True,
                       window_ms=100, n_reps=3, threshold_k=4.0, min_duration_ms=300,
                       min_gap_ms=300, max_onset_lookback_ms=300, max_offset_lookahead_ms=300,
                       max_pair_distance=None, show_pipeline_plots=False,
                       pipeline_margin_before_ms=500, pipeline_margin_after_ms=500,
                       pipeline_max_reps=None):
    """
    Deteta o onset (e, para condições isométricas, o offset) de EMG para
    ambos os músculos, em cada repetição, em cada condição pedida, usando o
    algoritmo `detect_onset_bp` (TKEO -> envolvente suavizada -> Hilbert ->
    mínimos locais da fase instantânea e da parte imaginária).

    Dois caminhos de deteção são usados automaticamente, dependendo de o
    DataFrame da condição ter ou não uma coluna 'Phase':
      - Se 'Phase' estiver presente (ex: HD), o início da fase concêntrica de
        cada repetição (partilhado entre músculos) é usado como referência
        para escolher, de entre todos os onsets devolvidos por
        `detect_onset_bp`, o mais próximo e anterior a essa referência (ver
        `_detect_emg_onsets_single_condition`).
      - Se 'Phase' estiver ausente (ex: HI, contrações isométricas), o onset e
        o offset são estimados diretamente a partir do sinal completo, de
        forma independente por músculo, e depois associados a cada
        repetição por proximidade a um plateau de ativação aproximado (ver
        `_detect_emg_onsets_isometric_condition`).

    :param subject_data: dict tal como produzido/consumido por
        `detect_concentric_eccentric_phases`.
    :param conditions: iterável com os nomes das condições a processar (ex:
        'HD', 'HI').
    :param order: ver `detect_onset_bp`.
    :param phase_sigma: ver `detect_onset_bp`.
    :param imag_sigma: ver `detect_onset_bp`.
    :param smooth_window: ver `detect_onset_bp`.
    :param emg_columns: nomes das colunas de EMG a processar. Por omissão,
        `EMG_COLUMNS` (EMG_Extensor_filt, EMG_Flexor_filt).
    :param show_plot: se True, faz o plot dos dois sinais de EMG com os
        onsets/offsets detetados, por condição.
    :param window_ms: janela (ms) da envolvente auxiliar usada apenas para
        localizar os plateaus de ativação (HI).
    :param n_reps: número esperado de contrações isométricas por músculo
        (apenas HI).
    :param threshold_k: multiplicador do limiar robusto (apenas HI).
    :param min_duration_ms: duração mínima de contração (ms) (apenas HI).
    :param min_gap_ms: intervalo máximo (ms) para fundir segmentos ativos
        próximos (apenas HI).
    :param max_onset_lookback_ms: janela máxima (ms) antes do ponto de
        referência (início da concêntrica em HD; início do plateau em HI)
        dentro da qual um onset ainda é aceite. Fora dessa janela, o onset
        fica a None em vez de "agarrar" um mínimo distante e provavelmente
        errado.
    :param max_offset_lookahead_ms: janela máxima (ms) depois do fim do
        plateau (apenas HI) dentro da qual um offset ainda é aceite.
    :param max_pair_distance: distância máxima (em amostras) entre um
        mínimo da fase e um mínimo da parte imaginária para serem
        considerados o mesmo evento e combinados por média (ver
        `detect_onset_bp` / `_pair_minima`). Por omissão (None), usa-se o
        próprio `order`. Isto substitui a combinação ingénua "1º com 1º, 2º
        com 2º" por posição no array, que desalinhava todos os onsets a
        partir do momento em que um dos dois sinais apanhasse um mínimo
        espúrio a mais ou a menos nalguma repetição anterior.
    :param show_pipeline_plots: se True, mostra adicionalmente os passos
        intermédios do algoritmo (EMG bruto, |TKEO|/envolvente suavizada,
        fase instantânea, parte imaginária) por repetição e por músculo,
        para inspeção visual do pipeline.
    :param pipeline_margin_before_ms: margem de sinal (ms) mostrada antes do
        onset, nos plots do pipeline.
    :param pipeline_margin_after_ms: margem de sinal (ms) mostrada depois do
        offset (ou do ponto de referência), nos plots do pipeline.
    :param pipeline_max_reps: se definido, limita o número de repetições
        mostradas por músculo, nos plots do pipeline.
    :return: tuplo (coord_hd, coord_hi), cada um um DataFrame pandas com uma
        linha por (repetição, músculo) -- ver `_results_to_dataframe`.
    """
    columns = emg_columns if emg_columns is not None else EMG_COLUMNS

    results_by_condition = {}
    for condition in conditions:
        if condition not in subject_data:
            print(f"Warning: condition '{condition}' does not exist in subject_data -- skipping.")
            continue

        df = subject_data[condition]['data']
        fs = subject_data[condition]['params']['sampling rate']

        if 'Phase' in df.columns:
            condition_results = _detect_emg_onsets_single_condition(
                df, fs, columns, order, phase_sigma, imag_sigma, smooth_window,
                max_onset_lookback_ms, max_pair_distance
            )
        else:
            condition_results = _detect_emg_onsets_isometric_condition(
                df, fs, columns, order, phase_sigma, imag_sigma, smooth_window,
                window_ms, n_reps, threshold_k, min_duration_ms, min_gap_ms,
                max_onset_lookback_ms, max_offset_lookahead_ms, max_pair_distance
            )

        results_by_condition[condition] = condition_results

        if show_plot:
            _plot_onsets(df, columns, condition_results, condition)

        if show_pipeline_plots:
            _plot_pipeline_steps_bp(
                df, columns, fs, phase_sigma, imag_sigma, smooth_window,
                condition_results, condition,
                margin_before_ms=pipeline_margin_before_ms,
                margin_after_ms=pipeline_margin_after_ms,
                max_reps=pipeline_max_reps,
            )

    results_dict = {condition: _results_to_dataframe(condition_results)
                    for condition, condition_results in results_by_condition.items()}

    info_hd = results_dict.get("HD", pd.DataFrame())
    info_hi = results_dict.get("HI", pd.DataFrame())

    return info_hd, info_hi


def _results_to_dataframe(results):
    """
    Converte uma lista de dicts de resultados por (repetição, músculo) num
    DataFrame pandas organizado.

    :param results: lista de dicts de resultados para uma única condição
        (possivelmente vazia).
    :return: pandas.DataFrame com uma linha por (repetição, músculo),
        ordenado por 'rep' e depois 'muscle'. DataFrame vazio se `results`
        estiver vazio.
    """
    df = pd.DataFrame(results)
    if df.empty:
        return df
    return df.sort_values(['rep', 'muscle']).reset_index(drop=True)


def _plot_pipeline_steps_bp(df, columns, fs, phase_sigma, imag_sigma, smooth_window,
                             results, condition, margin_before_ms=500, margin_after_ms=500,
                             max_reps=None):
    """
    Faz o plot, para cada repetição e músculo, dos passos intermédios do
    algoritmo `detect_onset_bp`: o EMG bruto, a envolvente |TKEO| (em bruto e
    suavizada), a fase instantânea suavizada e a parte imaginária suavizada
    -- com o ponto de referência, o onset e (quando disponível) o offset
    marcados em todos os painéis, para inspeção visual do pipeline.

    :param df: DataFrame com as colunas de EMG a plotar.
    :param columns: lista de nomes das colunas de EMG.
    :param fs: frequência de amostragem (Hz).
    :param phase_sigma: ver `detect_onset_bp`.
    :param imag_sigma: ver `detect_onset_bp`.
    :param smooth_window: ver `detect_onset_bp`.
    :param results: lista de dicts de resultados de onset/offset.
    :param condition: nome da condição (usado no título do gráfico).
    :param margin_before_ms: quanto sinal (ms) mostrar antes do onset.
    :param margin_after_ms: quanto sinal (ms) mostrar depois do offset (ou do
        ponto de referência, se não houver offset).
    :param max_reps: se definido, limita o número de repetições plotadas por
        músculo.
    :return: None. Mostra uma figura por (músculo, repetição).
    """
    margin_before_samples = int(round((margin_before_ms / 1000.0) * fs))
    margin_after_samples = int(round((margin_after_ms / 1000.0) * fs))

    for col in columns:
        raw = df[col].to_numpy()
        x_feat, x_smooth, ins_phase_s, imag_part_s = _bp_pipeline_components(
            raw, phase_sigma, imag_sigma, smooth_window
        )

        muscle_results = [r for r in results if r['muscle'] == col]
        if max_reps is not None:
            muscle_results = muscle_results[:max_reps]

        for r in muscle_results:
            ref_idx = r['concentric_start_idx']
            onset_idx = r['onset_idx']
            offset_idx = r.get('offset_idx')
            if offset_idx is None:
                offset_idx = r.get('concentric_end_idx', ref_idx)

            anchor_start = onset_idx if onset_idx is not None else ref_idx
            anchor_end = offset_idx if offset_idx is not None else ref_idx

            plot_start = max(0, anchor_start - margin_before_samples)
            plot_end = min(raw.size, anchor_end + margin_after_samples)
            sl = slice(plot_start, plot_end)
            t = np.arange(plot_start, plot_end)

            fig, axes = plt.subplots(4, 1, figsize=(10, 7.5), sharex=True)

            axes[0].plot(t, raw[sl], color='black', linewidth=0.8)
            axes[0].set_ylabel('EMG bruto', fontsize=8)

            axes[1].plot(t, x_feat[sl], color='tab:gray', linewidth=0.6, alpha=0.7, label='|TKEO|')
            axes[1].plot(t, x_smooth[sl], color='tab:orange', linewidth=1.6, label='envolvente suavizada')
            axes[1].set_ylabel('|TKEO| /\nenvolvente', fontsize=8)
            axes[1].legend(fontsize=7, loc='upper right')

            axes[2].plot(t, ins_phase_s[sl], color='tab:blue', linewidth=1.2)
            axes[2].set_ylabel('Fase instantânea\n(suavizada)', fontsize=8)

            axes[3].plot(t, imag_part_s[sl], color='tab:green', linewidth=1.2)
            axes[3].set_ylabel('Parte imaginária\n(suavizada)', fontsize=8)

            for ax in axes:
                ax.axvline(ref_idx, color='tab:blue', linestyle='--', alpha=0.6)
                if onset_idx is not None:
                    ax.axvline(onset_idx, color='tab:red', linestyle='-', alpha=0.8)
                if offset_idx is not None:
                    ax.axvline(offset_idx, color='tab:green', linestyle='-', alpha=0.8)

            axes[-1].set_xlabel('Amostras')
            fig.suptitle(
                f"{condition} -- {col} -- Rep {r['rep']} -- passos do pipeline (detect_onset_bp)\n"
                f"(azul tracejado = ponto de referência, vermelho = onset, verde = offset "
                f"ou fim da fase concêntrica)",
                fontsize=10,
            )
            plt.tight_layout()
            plt.show()


def _plot_onsets(df, columns, results, condition):
    """
    Faz o plot de cada canal de EMG com os onsets, offsets (quando
    disponíveis) e pontos de referência detetados.

    :param df: DataFrame com as colunas de EMG a plotar.
    :param columns: lista de nomes das colunas de EMG.
    :param results: lista de dicts de resultados de onset/offset.
    :param condition: nome da condição (usado no título do gráfico).
    :return: None. Mostra o gráfico.
    """
    n = len(columns)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        signal = df[col].to_numpy()
        ax.plot(signal, color='black', linewidth=0.8)

        col_results = [r for r in results if r['muscle'] == col]
        for i, r in enumerate(col_results):
            show_label = i == 0
            ax.axvline(r['concentric_start_idx'], color='tab:blue', linestyle='--', alpha=0.6,
                       label='Reference point' if show_label else None)
            if r['onset_idx'] is not None:
                ax.plot(r['onset_idx'], signal[r['onset_idx']], 'r^', markersize=8,
                        label='EMG onset' if show_label else None)
            if r.get('offset_idx') is not None:
                ax.plot(r['offset_idx'], signal[r['offset_idx']], 'gv', markersize=8,
                        label='EMG offset' if show_label else None)

        ax.set_ylabel(col, fontsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=8)
    axes[-1].set_xlabel('Samples')
    fig.suptitle(f'EMG Onset/Offset Detection (detect_onset_bp) -- {condition} Condition')
    plt.tight_layout()
    plt.show()