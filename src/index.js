const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Admin-Code",
  "Access-Control-Allow-Methods": "GET,POST,DELETE,OPTIONS",
};

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: CORS_HEADERS,
      });
    }

    const url = new URL(request.url);

    if (request.method === "POST" && url.pathname === "/auth/request-code") {
      return postAuthRequestCode(request, env);
    }

    if (request.method === "POST" && url.pathname === "/auth/verify-code") {
      return postAuthVerifyCode(request, env);
    }

    if (request.method === "GET" && url.pathname === "/auth/me") {
      return getAuthMe(request, env);
    }

    if (request.method === "POST" && url.pathname === "/auth/logout") {
      return postAuthLogout(request, env);
    }

    if (request.method === "GET" && url.pathname === "/jars") {
      return getJars(request, env);
    }

    if (request.method === "GET" && url.pathname === "/jar-requests") {
      return getJarRequests(request, env);
    }

    if (request.method === "POST" && url.pathname === "/jar-requests") {
      return postJarRequest(request, env);
    }

    if (request.method === "POST" && /^\/jar-requests\/\d+\/approve$/.test(url.pathname)) {
      return postApproveJarRequest(request, url, env);
    }

    if (request.method === "POST" && /^\/jar-requests\/\d+\/reject$/.test(url.pathname)) {
      return postRejectJarRequest(request, url, env);
    }

    if (request.method === "POST" && url.pathname === "/jars") {
      return postJars(request, env);
    }

    if (request.method === "DELETE" && /^\/jars\/\d+$/.test(url.pathname)) {
      return deleteJar(request, url, env);
    }

    if (request.method === "GET" && url.pathname === "/wallet") {
      return getWallet(request, url, env);
    }

    if (request.method === "POST" && url.pathname === "/devices/register") {
      return postDevicesRegister(request, env);
    }

    if (request.method === "POST" && url.pathname === "/devices/unregister") {
      return postDevicesUnregister(request, env);
    }

    if (request.method === "POST" && url.pathname === "/notifications/publish-latest") {
      return postPublishLatestNotification(request, env);
    }

    if (request.method === "POST" && url.pathname === "/wallet/topup") {
      return postWalletTopup(request, env);
    }

    if (request.method === "POST" && url.pathname === "/contribute") {
      return postContribute(request, env);
    }

    return env.ASSETS.fetch(request);
  },
};

async function postAuthRequestCode(request, env) {
  if (!env.RESEND_API_KEY || !env.AUTH_FROM_EMAIL) {
    return jsonResponse(
      {
        error:
          "이메일 발송 설정이 아직 끝나지 않았어요. Resend API 키와 발신 이메일을 먼저 연결해 주세요.",
      },
      { status: 503 },
    );
  }

  let body;

  try {
    body = await request.json();
  } catch {
    return jsonResponse(
      { error: "Request body must be valid JSON." },
      { status: 400 },
    );
  }

  const email = normalizeEmail(body.email);
  if (!isValidEmail(email)) {
    return jsonResponse(
      { error: "이메일 형식이 맞지 않아요." },
      { status: 400 },
    );
  }

  const latestCode = await env.DB.prepare(
    `
      SELECT created_at
      FROM email_login_codes
      WHERE email = ?
      ORDER BY id DESC
      LIMIT 1
    `,
  )
    .bind(email)
    .first();

  if (
    latestCode?.created_at &&
    Date.now() - Date.parse(latestCode.created_at) < 60 * 1000
  ) {
    return jsonResponse(
      { error: "인증코드는 1분에 한 번만 다시 보낼 수 있어요." },
      { status: 429 },
    );
  }

  const code = generateSixDigitCode();
  const ttlMinutes = Number.parseInt(env.AUTH_CODE_TTL_MINUTES ?? "10", 10);
  const expiresAt = new Date(
    Date.now() + ttlMinutes * 60 * 1000,
  ).toISOString();

  await env.DB.prepare(
    `
      INSERT INTO email_login_codes (email, code, expires_at)
      VALUES (?, ?, ?)
    `,
  )
    .bind(email, code, expiresAt)
    .run();

  const subjectPrefix = env.AUTH_EMAIL_SUBJECT_PREFIX || "Maka Workshop";
  const resendResponse = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.RESEND_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      from: env.AUTH_FROM_EMAIL,
      to: [email],
      subject: `[${subjectPrefix}] 커피공방 인증코드 ${code}`,
      text: buildAuthEmailText(code, ttlMinutes),
      html: buildAuthEmailHtml(code, ttlMinutes),
    }),
  });

  if (!resendResponse.ok) {
    const errorText = await resendResponse.text();
    console.error("Failed to send auth email", errorText);

    return jsonResponse(
      { error: "인증 메일을 보내지 못했어요. 메일 설정을 확인해 주세요." },
      { status: 502 },
    );
  }

  return jsonResponse({
    ok: true,
    message: "인증코드를 이메일로 보냈어요. 메일함과 스팸함을 확인해 주세요.",
  });
}

async function postAuthVerifyCode(request, env) {
  let body;

  try {
    body = await request.json();
  } catch {
    return jsonResponse(
      { error: "Request body must be valid JSON." },
      { status: 400 },
    );
  }

  const email = normalizeEmail(body.email);
  const code = String(body.code ?? "").trim();

  if (!isValidEmail(email)) {
    return jsonResponse(
      { error: "이메일 형식이 맞지 않아요." },
      { status: 400 },
    );
  }

  if (!/^\d{6}$/.test(code)) {
    return jsonResponse(
      { error: "6자리 인증코드를 입력해 주세요." },
      { status: 400 },
    );
  }

  const loginCode = await env.DB.prepare(
    `
      SELECT id, email, code, expires_at, used_at
      FROM email_login_codes
      WHERE email = ? AND code = ?
      ORDER BY id DESC
      LIMIT 1
    `,
  )
    .bind(email, code)
    .first();

  if (!loginCode || loginCode.used_at) {
    return jsonResponse(
      { error: "인증코드가 맞지 않아요." },
      { status: 400 },
    );
  }

  if (Date.parse(loginCode.expires_at) < Date.now()) {
    return jsonResponse(
      { error: "인증코드가 만료되었어요. 다시 요청해 주세요." },
      { status: 400 },
    );
  }

  await env.DB.prepare(
    `
      UPDATE email_login_codes
      SET used_at = CURRENT_TIMESTAMP
      WHERE id = ?
    `,
  )
    .bind(loginCode.id)
    .run();

  let user = await env.DB.prepare(
    `
      SELECT u.id, u.name, ue.email
      FROM user_emails ue
      JOIN users u ON u.id = ue.user_id
      WHERE ue.email = ?
      LIMIT 1
    `,
  )
    .bind(email)
    .first();

  if (!user) {
    const fallbackName = displayNameFromEmail(email);

    const createdUser = await env.DB.prepare(
      `
        INSERT INTO users (name)
        VALUES (?)
      `,
    )
      .bind(fallbackName)
      .run();

    const userId = createdUser.meta?.last_row_id;

    await env.DB.prepare(
      `
        INSERT INTO wallets (user_id, bean_balance)
        VALUES (?, 0)
      `,
    )
      .bind(userId)
      .run();

    await env.DB.prepare(
      `
        INSERT INTO user_emails (user_id, email, verified_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
      `,
    )
      .bind(userId, email)
      .run();

    user = {
      id: userId,
      name: fallbackName,
      email,
    };
  }

  const sessionDays = Number.parseInt(env.AUTH_SESSION_DAYS ?? "30", 10);
  const rawToken = generateSessionToken();
  const tokenHash = await sha256Hex(rawToken);
  const expiresAt = new Date(
    Date.now() + sessionDays * 24 * 60 * 60 * 1000,
  ).toISOString();

  await env.DB.prepare(
    `
      INSERT INTO sessions (user_id, token_hash, expires_at)
      VALUES (?, ?, ?)
    `,
  )
    .bind(user.id, tokenHash, expiresAt)
    .run();

  return jsonResponse({
    ok: true,
    user,
    session: {
      token: rawToken,
      expires_at: expiresAt,
    },
  });
}

async function getAuthMe(request, env) {
  const authUser = await getAuthenticatedUser(request, env);

  if (!authUser) {
    return jsonResponse(
      { error: "로그인이 필요해요." },
      { status: 401 },
    );
  }

  return jsonResponse({
    ok: true,
    user: {
      id: authUser.id,
      name: authUser.name,
      email: authUser.email,
    },
  });
}

async function postAuthLogout(request, env) {
  const authHeader = request.headers.get("Authorization");
  const token = extractBearerToken(authHeader);

  if (!token) {
    return jsonResponse(
      { error: "로그인이 필요해요." },
      { status: 401 },
    );
  }

  const tokenHash = await sha256Hex(token);

  await env.DB.prepare(
    `
      UPDATE sessions
      SET revoked_at = CURRENT_TIMESTAMP
      WHERE token_hash = ?
    `,
  )
    .bind(tokenHash)
    .run();

  return jsonResponse({
    ok: true,
    message: "로그아웃되었어요.",
  });
}

async function getJars(request, env) {
  const authUser = await getAuthenticatedUser(request, env);
  const userId = authUser?.id ?? 1;

  const result = await env.DB.prepare(
    `
      SELECT
        j.id,
        j.ticker,
        j.display_name,
        j.note,
        j.comment,
        j.total_beans,
        COALESCE(SUM(CASE WHEN c.user_id = ? THEN c.amount ELSE 0 END), 0) AS my_contribution
      FROM jars j
      LEFT JOIN contributions c ON c.jar_id = j.id
      GROUP BY j.id, j.ticker, j.display_name, j.note, j.comment, j.total_beans
      ORDER BY
        CASE WHEN j.ticker = 'Maka' THEN 0 ELSE 1 END,
        j.total_beans DESC,
        j.ticker ASC
    `,
  )
    .bind(userId)
    .all();

  return jsonResponse({
    jars: result.results ?? [],
  });
}

async function getJarRequests(request, env) {
  if (!isValidAdminRequest(request, env)) {
    return jsonResponse(
      { error: "관리자만 병 요청 목록을 볼 수 있어요." },
      { status: 403 },
    );
  }

  const result = await env.DB.prepare(
    `
      SELECT
        jr.id,
        jr.ticker,
        jr.status,
        u.name AS requester_name,
        ue.email AS requester_email,
        jr.created_at
      FROM jar_requests jr
      JOIN users u ON u.id = jr.requested_by_user_id
      LEFT JOIN user_emails ue ON ue.user_id = u.id
      WHERE jr.status = 'pending'
      ORDER BY jr.created_at ASC, jr.id ASC
    `,
  ).all();

  return jsonResponse({
    requests: result.results ?? [],
  });
}

async function postJarRequest(request, env) {
  let body;

  try {
    body = await request.json();
  } catch {
    return jsonResponse(
      { error: "Request body must be valid JSON." },
      { status: 400 },
    );
  }

  const authUser = await getAuthenticatedUser(request, env);

  if (!authUser) {
    return jsonResponse(
      { error: "로그인 후 병을 추가할 수 있어요." },
      { status: 401 },
    );
  }

  const ticker = normalizeJarTicker(body.ticker);

  if (!ticker) {
    return jsonResponse(
      { error: "티커를 입력해 주세요." },
      { status: 400 },
    );
  }

  const existingJar = await findJarByTicker(ticker, env);
  if (existingJar) {
    return jsonResponse(
      { error: "이미 같은 이름의 병이 있어요." },
      { status: 409 },
    );
  }

  const existingRequest = await env.DB.prepare(
    `
      SELECT id
      FROM jar_requests
      WHERE ticker = ? AND status = 'pending'
      LIMIT 1
    `,
  )
    .bind(ticker)
    .first();

  if (existingRequest) {
    return jsonResponse(
      { error: "이미 승인 대기 중인 병 요청이 있어요." },
      { status: 409 },
    );
  }

  const insertResult = await env.DB.prepare(
    `
      INSERT INTO jar_requests (ticker, requested_by_user_id, status)
      VALUES (?, ?, 'pending')
    `,
  )
    .bind(ticker, authUser.id)
    .run();

  const createdRequest = await env.DB.prepare(
    `
      SELECT
        jr.id,
        jr.ticker,
        jr.status,
        u.name AS requester_name,
        ue.email AS requester_email,
        jr.created_at
      FROM jar_requests jr
      JOIN users u ON u.id = jr.requested_by_user_id
      LEFT JOIN user_emails ue ON ue.user_id = u.id
      WHERE jr.id = ?
      LIMIT 1
    `,
  )
    .bind(insertResult.meta?.last_row_id)
    .first();

  return jsonResponse({
    ok: true,
    message: "병 추가 요청을 보냈어요. 관리자 승인 후 반영됩니다.",
    request: createdRequest,
  });
}

async function postApproveJarRequest(request, url, env) {
  if (!isValidAdminRequest(request, env)) {
    return jsonResponse(
      { error: "관리자만 병 요청을 승인할 수 있어요." },
      { status: 403 },
    );
  }

  const requestId = parsePositiveInteger(url.pathname.split("/")[2]);
  if (!Number.isInteger(requestId) || requestId <= 0) {
    return jsonResponse(
      { error: "request_id must be a positive integer." },
      { status: 400 },
    );
  }

  const jarRequest = await env.DB.prepare(
    `
      SELECT id, ticker, status
      FROM jar_requests
      WHERE id = ?
      LIMIT 1
    `,
  )
    .bind(requestId)
    .first();

  if (!jarRequest) {
    return jsonResponse({ error: "요청을 찾지 못했어요." }, { status: 404 });
  }

  if (jarRequest.status !== "pending") {
    return jsonResponse(
      { error: "이미 처리된 요청이에요." },
      { status: 400 },
    );
  }

  const existingJar = await findJarByTicker(jarRequest.ticker, env);
  if (existingJar) {
    await env.DB.prepare(
      `
        UPDATE jar_requests
        SET status = 'merged',
            reviewed_at = CURRENT_TIMESTAMP
        WHERE id = ?
      `,
    )
      .bind(requestId)
      .run();

    return jsonResponse({
      ok: true,
      message: "이미 있는 병이라 요청을 병합 처리했어요.",
      jar: existingJar,
    });
  }

  const jar = await createJarRecord(jarRequest.ticker, env);

  await env.DB.prepare(
    `
      UPDATE jar_requests
      SET status = 'approved',
          approved_jar_id = ?,
          reviewed_at = CURRENT_TIMESTAMP
      WHERE id = ?
    `,
  )
    .bind(jar.id, requestId)
    .run();

  return jsonResponse({
    ok: true,
    message: `${jar.ticker} 병 요청을 승인했어요.`,
    jar,
  });
}

async function postRejectJarRequest(request, url, env) {
  if (!isValidAdminRequest(request, env)) {
    return jsonResponse(
      { error: "관리자만 병 요청을 반려할 수 있어요." },
      { status: 403 },
    );
  }

  const requestId = parsePositiveInteger(url.pathname.split("/")[2]);
  if (!Number.isInteger(requestId) || requestId <= 0) {
    return jsonResponse(
      { error: "request_id must be a positive integer." },
      { status: 400 },
    );
  }

  const updateResult = await env.DB.prepare(
    `
      UPDATE jar_requests
      SET status = 'rejected',
          reviewed_at = CURRENT_TIMESTAMP
      WHERE id = ? AND status = 'pending'
    `,
  )
    .bind(requestId)
    .run();

  if ((updateResult.meta?.changes ?? 0) !== 1) {
    return jsonResponse(
      { error: "처리할 승인 대기 요청이 없어요." },
      { status: 404 },
    );
  }

  return jsonResponse({
    ok: true,
    message: "병 요청을 반려했어요.",
  });
}

async function postJars(request, env) {
  let body;

  try {
    body = await request.json();
  } catch {
    return jsonResponse(
      { error: "Request body must be valid JSON." },
      { status: 400 },
    );
  }

  const isAdmin = isValidAdminRequest(request, env);

  if (!isAdmin) {
    return jsonResponse(
      { error: "관리자만 병을 바로 추가할 수 있어요." },
      { status: 403 },
    );
  }

  const ticker = normalizeJarTicker(body.ticker);

  if (!ticker) {
    return jsonResponse(
      { error: "티커를 입력해 주세요." },
      { status: 400 },
    );
  }

  const existingJar = await findJarByTicker(ticker, env);

  if (existingJar) {
    return jsonResponse(
      { error: "이미 같은 이름의 병이 있어요." },
      { status: 409 },
    );
  }

  try {
    const jar = await createJarRecord(ticker, env);
    return jsonResponse({
      ok: true,
      message: "병이 추가되었어요.",
      jar,
    });
  } catch (error) {
    console.error("Failed to create jar", error);

    return jsonResponse(
      { error: "병을 추가하지 못했어요." },
      { status: 500 },
    );
  }

  return jsonResponse(
    { error: "병을 추가하지 못했어요." },
    { status: 500 },
  );
}

async function deleteJar(request, url, env) {
  if (!isValidAdminRequest(request, env)) {
    return jsonResponse(
      { error: "관리자만 병을 삭제할 수 있어요." },
      { status: 403 },
    );
  }

  const jarId = parsePositiveInteger(url.pathname.split("/").pop());

  if (!Number.isInteger(jarId) || jarId <= 0) {
    return jsonResponse(
      { error: "jar_id must be a positive integer." },
      { status: 400 },
    );
  }

  const jar = await env.DB.prepare(
    `
      SELECT
        id,
        ticker,
        display_name,
        note,
        comment,
        total_beans,
        0 AS my_contribution
      FROM jars
      WHERE id = ?
      LIMIT 1
    `,
  )
    .bind(jarId)
    .first();

  if (!jar) {
    return jsonResponse({ error: "Jar not found." }, { status: 404 });
  }

  if (jar.ticker === "Maka") {
    return jsonResponse(
      { error: "Maka 병은 삭제할 수 없어요." },
      { status: 400 },
    );
  }

  const deleteResult = await env.DB.prepare(
    `
      DELETE FROM jars
      WHERE id = ?
    `,
  )
    .bind(jarId)
    .run();

  if ((deleteResult.meta?.changes ?? 0) !== 1) {
    return jsonResponse(
      { error: "병을 삭제하지 못했어요." },
      { status: 500 },
    );
  }

  return jsonResponse({
    ok: true,
    message: "병을 삭제했어요.",
    jar,
  });
}

async function getWallet(request, url, env) {
  const authUser = await getAuthenticatedUser(request, env);
  const userId = authUser?.id ?? parsePositiveInteger(url.searchParams.get("user_id"));

  if (!Number.isInteger(userId) || userId <= 0) {
    return jsonResponse(
      { error: "user_id must be a positive integer." },
      { status: 400 },
    );
  }

  const wallet = await fetchWalletByUserId(userId, env);

  if (!wallet) {
    return jsonResponse({ error: "Wallet not found." }, { status: 404 });
  }

  return jsonResponse(wallet);
}

async function postDevicesRegister(request, env) {
  let body;

  try {
    body = await request.json();
  } catch {
    return jsonResponse(
      { error: "Request body must be valid JSON." },
      { status: 400 },
    );
  }

  const token = String(body.token ?? "").trim();
  const platform = String(body.platform ?? "ios").trim().toLowerCase();
  const environment = String(body.environment ?? "sandbox").trim().toLowerCase();
  const authUser = await getAuthenticatedUser(request, env);

  if (!/^[a-f0-9]{64,}$/i.test(token)) {
    return jsonResponse(
      { error: "유효한 device token이 아니에요." },
      { status: 400 },
    );
  }

  if (!["ios"].includes(platform)) {
    return jsonResponse(
      { error: "지원하지 않는 플랫폼입니다." },
      { status: 400 },
    );
  }

  if (!["sandbox", "production"].includes(environment)) {
    return jsonResponse(
      { error: "지원하지 않는 푸시 환경입니다." },
      { status: 400 },
    );
  }

  await env.DB.prepare(
    `
      INSERT INTO device_tokens (
        token,
        platform,
        environment,
        user_id,
        is_active,
        last_registered_at,
        last_error
      )
      VALUES (?, ?, ?, ?, 1, CURRENT_TIMESTAMP, NULL)
      ON CONFLICT(token) DO UPDATE SET
        platform = excluded.platform,
        environment = excluded.environment,
        user_id = COALESCE(excluded.user_id, device_tokens.user_id),
        is_active = 1,
        last_registered_at = CURRENT_TIMESTAMP,
        last_error = NULL
    `,
  )
    .bind(token, platform, environment, authUser?.id ?? null)
    .run();

  return jsonResponse({
    ok: true,
    message: "기기 알림 토큰을 등록했어요.",
  });
}

async function postDevicesUnregister(request, env) {
  let body;

  try {
    body = await request.json();
  } catch {
    return jsonResponse(
      { error: "Request body must be valid JSON." },
      { status: 400 },
    );
  }

  const token = String(body.token ?? "").trim();

  if (!token) {
    return jsonResponse(
      { error: "device token이 필요해요." },
      { status: 400 },
    );
  }

  await env.DB.prepare(
    `
      UPDATE device_tokens
      SET is_active = 0,
          last_error = 'unregistered'
      WHERE token = ?
    `,
  )
    .bind(token)
    .run();

  return jsonResponse({
    ok: true,
    message: "기기 알림 토큰을 비활성화했어요.",
  });
}

async function postPublishLatestNotification(request, env) {
  if (!isValidAdminRequest(request, env)) {
    return jsonResponse(
      { error: "관리자만 새 글 푸시를 보낼 수 있어요." },
      { status: 403 },
    );
  }

  if (!env.APNS_PRIVATE_KEY || !env.APNS_KEY_ID || !env.APNS_TEAM_ID || !env.APNS_BUNDLE_ID) {
    return jsonResponse(
      { error: "APNs 설정이 아직 끝나지 않았어요." },
      { status: 503 },
    );
  }

  const latest = await readJsonAsset("/data/latest.json", request.url, env);
  const metaPath = String(latest.meta_path ?? "").trim();

  if (!metaPath) {
    return jsonResponse(
      { error: "latest.json에 meta_path가 없어요." },
      { status: 500 },
    );
  }

  const normalizedMetaPath = metaPath.startsWith("/") ? metaPath : `/${metaPath.replace(/^web\//, "")}`;
  const meta = await readJsonAsset(normalizedMetaPath, request.url, env);
  const title = extractLatestPushTitle(meta);
  const body = extractLatestPushBody(meta);
  const eventKey = normalizedMetaPath;

  const existingEvent = await env.DB.prepare(
    `
      SELECT id
      FROM post_push_events
      WHERE event_key = ?
      LIMIT 1
    `,
  )
    .bind(eventKey)
    .first();

  if (existingEvent) {
    return jsonResponse({
      ok: true,
      message: "이미 이 글에 대한 푸시를 보냈어요.",
      sent_count: 0,
      skipped: true,
    });
  }

  const devicesResult = await env.DB.prepare(
    `
      SELECT token, environment
      FROM device_tokens
      WHERE is_active = 1
        AND platform = 'ios'
    `,
  ).all();

  const devices = devicesResult.results ?? [];
  let sentCount = 0;
  let failedCount = 0;

  for (const device of devices) {
    const pushResult = await sendApnsNotification(
      {
        token: device.token,
        environment: device.environment,
        title,
        body,
      },
      env,
    );

    if (pushResult.ok) {
      sentCount += 1;
      await env.DB.prepare(
        `
          UPDATE device_tokens
          SET last_notified_at = CURRENT_TIMESTAMP,
              last_error = NULL
          WHERE token = ?
        `,
      )
        .bind(device.token)
        .run();
    } else {
      failedCount += 1;
      const shouldDeactivate = pushResult.status === 410 || pushResult.status === 400;

      await env.DB.prepare(
        `
          UPDATE device_tokens
          SET is_active = CASE WHEN ? THEN 0 ELSE is_active END,
              last_error = ?
          WHERE token = ?
        `,
      )
        .bind(shouldDeactivate ? 1 : 0, pushResult.error ?? `APNs ${pushResult.status}`, device.token)
        .run();
    }
  }

  await env.DB.prepare(
    `
      INSERT INTO post_push_events (event_key, title, body, sent_count)
      VALUES (?, ?, ?, ?)
    `,
  )
    .bind(eventKey, title, body, sentCount)
    .run();

  return jsonResponse({
    ok: true,
    message: "최신 글 푸시 발송을 마쳤어요.",
    sent_count: sentCount,
    failed_count: failedCount,
    title,
    body,
  });
}

async function postWalletTopup(request, env) {
  let body;

  try {
    body = await request.json();
  } catch {
    return jsonResponse(
      { error: "Request body must be valid JSON." },
      { status: 400 },
    );
  }

  const authUser = await getAuthenticatedUser(request, env);
  const userId = authUser?.id ?? parsePositiveInteger(body.user_id);
  const amount = parsePositiveInteger(body.amount);

  if (!Number.isInteger(userId) || userId <= 0) {
    return jsonResponse(
      { error: "user_id must be a positive integer." },
      { status: 400 },
    );
  }

  if (!Number.isInteger(amount) || amount <= 0) {
    return jsonResponse(
      { error: "amount must be a positive integer." },
      { status: 400 },
    );
  }

  const walletUpdate = await env.DB.prepare(
    `
      UPDATE wallets
      SET bean_balance = bean_balance + ?,
          updated_at = CURRENT_TIMESTAMP
      WHERE user_id = ?
    `,
  )
    .bind(amount, userId)
    .run();

  if ((walletUpdate.meta?.changes ?? 0) !== 1) {
    return jsonResponse({ error: "Wallet not found." }, { status: 404 });
  }

  const updatedWallet = await fetchWalletByUserId(userId, env);

  if (!updatedWallet) {
    return jsonResponse({ error: "Wallet not found." }, { status: 404 });
  }

  let topupRecord;

  try {
    const topupInsert = await env.DB.prepare(
      `
        INSERT INTO wallet_topups (user_id, amount, balance_after)
        VALUES (?, ?, ?)
      `,
    )
      .bind(userId, amount, updatedWallet.bean_balance)
      .run();

    topupRecord = await env.DB.prepare(
      `
        SELECT id, user_id, amount, balance_after, source, created_at
        FROM wallet_topups
        WHERE id = ?
      `,
    )
      .bind(topupInsert.meta?.last_row_id)
      .first();
  } catch (error) {
    await env.DB.prepare(
      `
        UPDATE wallets
        SET bean_balance = bean_balance - ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE user_id = ? AND bean_balance >= ?
      `,
    )
      .bind(amount, userId, amount)
      .run();

    console.error("Failed to save wallet topup", error);

    return jsonResponse(
      { error: "커피콩 구매 내역을 저장하지 못했어요." },
      { status: 500 },
    );
  }

  return jsonResponse({
    ok: true,
    message: "Wallet topped up.",
    wallet: updatedWallet,
    topup: topupRecord,
  });
}

async function postContribute(request, env) {
  let body;

  try {
    body = await request.json();
  } catch {
    return jsonResponse(
      { error: "Request body must be valid JSON." },
      { status: 400 },
    );
  }

  const authUser = await getAuthenticatedUser(request, env);
  const userId = authUser?.id ?? parsePositiveInteger(body.user_id);
  const jarId = parsePositiveInteger(body.jar_id);
  const amount = parsePositiveInteger(body.amount);

  if (!Number.isInteger(userId) || userId <= 0) {
    return jsonResponse(
      { error: "user_id must be a positive integer." },
      { status: 400 },
    );
  }

  if (!Number.isInteger(jarId) || jarId <= 0) {
    return jsonResponse(
      { error: "jar_id must be a positive integer." },
      { status: 400 },
    );
  }

  if (!Number.isInteger(amount) || amount <= 0) {
    return jsonResponse(
      { error: "amount must be a positive integer." },
      { status: 400 },
    );
  }

  const wallet = await env.DB.prepare(
    "SELECT bean_balance FROM wallets WHERE user_id = ?",
  )
    .bind(userId)
    .first();

  if (!wallet) {
    return jsonResponse({ error: "Wallet not found." }, { status: 404 });
  }

  if (wallet.bean_balance < amount) {
    return jsonResponse(
      { error: "Not enough beans in wallet." },
      { status: 400 },
    );
  }

  const jar = await env.DB.prepare(
    "SELECT id, ticker, total_beans FROM jars WHERE id = ?",
  )
    .bind(jarId)
    .first();

  if (!jar) {
    return jsonResponse({ error: "Jar not found." }, { status: 404 });
  }

  const walletUpdate = await env.DB.prepare(
    `
      UPDATE wallets
      SET bean_balance = bean_balance - ?,
          updated_at = CURRENT_TIMESTAMP
      WHERE user_id = ? AND bean_balance >= ?
    `,
  )
    .bind(amount, userId, amount)
    .run();

  if ((walletUpdate.meta?.changes ?? 0) !== 1) {
    return jsonResponse(
      { error: "Not enough beans in wallet." },
      { status: 400 },
    );
  }

  let jarWasUpdated = false;

  try {
    const jarUpdate = await env.DB.prepare(
      "UPDATE jars SET total_beans = total_beans + ? WHERE id = ?",
    )
      .bind(amount, jarId)
      .run();

    if ((jarUpdate.meta?.changes ?? 0) !== 1) {
      throw new Error("Failed to update jar.");
    }

    jarWasUpdated = true;

    await env.DB.prepare(
      "INSERT INTO contributions (user_id, jar_id, amount) VALUES (?, ?, ?)",
    )
      .bind(userId, jarId, amount)
      .run();
  } catch (error) {
    await env.DB.prepare(
      `
        UPDATE wallets
        SET bean_balance = bean_balance + ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE user_id = ?
      `,
    )
      .bind(amount, userId)
      .run();

    if (jarWasUpdated) {
      await env.DB.prepare(
        "UPDATE jars SET total_beans = total_beans - ? WHERE id = ? AND total_beans >= ?",
      )
        .bind(amount, jarId, amount)
        .run();
    }

    console.error("Failed to save contribution", error);

    return jsonResponse(
      { error: "Failed to save contribution." },
      { status: 500 },
    );
  }

  const updatedWallet = await env.DB.prepare(
    "SELECT user_id, bean_balance FROM wallets WHERE user_id = ?",
  )
    .bind(userId)
    .first();

  const updatedJar = await env.DB.prepare(
    "SELECT id, ticker, total_beans FROM jars WHERE id = ?",
  )
    .bind(jarId)
    .first();

  return jsonResponse({
    ok: true,
    message: "Contribution saved.",
    contribution: {
      user_id: userId,
      jar_id: jarId,
      amount,
    },
    wallet: updatedWallet,
    jar: updatedJar,
  });
}

async function fetchWalletByUserId(userId, env) {
  return env.DB.prepare(
    `
      SELECT
        u.id AS user_id,
        u.name,
        w.bean_balance
      FROM wallets w
      JOIN users u ON u.id = w.user_id
      WHERE w.user_id = ?
    `,
  )
    .bind(userId)
    .first();
}

async function getAuthenticatedUser(request, env) {
  const token = extractBearerToken(request.headers.get("Authorization"));
  if (!token) {
    return null;
  }

  const tokenHash = await sha256Hex(token);
  const session = await env.DB.prepare(
    `
      SELECT
        s.id AS session_id,
        s.user_id,
        s.expires_at,
        u.name,
        ue.email
      FROM sessions s
      JOIN users u ON u.id = s.user_id
      JOIN user_emails ue ON ue.user_id = u.id
      WHERE s.token_hash = ?
        AND s.revoked_at IS NULL
      LIMIT 1
    `,
  )
    .bind(tokenHash)
    .first();

  if (!session) {
    return null;
  }

  if (Date.parse(session.expires_at) < Date.now()) {
    await env.DB.prepare(
      `
        UPDATE sessions
        SET revoked_at = CURRENT_TIMESTAMP
        WHERE id = ?
      `,
    )
      .bind(session.session_id)
      .run();

    return null;
  }

  await env.DB.prepare(
    `
      UPDATE sessions
      SET last_used_at = CURRENT_TIMESTAMP
      WHERE id = ?
    `,
  )
    .bind(session.session_id)
    .run();

  return {
    id: session.user_id,
    name: session.name,
    email: session.email,
    sessionId: session.session_id,
  };
}

async function findJarByTicker(ticker, env) {
  return env.DB.prepare(
    `
      SELECT
        id,
        ticker,
        display_name,
        note,
        comment,
        total_beans,
        0 AS my_contribution
      FROM jars
      WHERE ticker = ?
      LIMIT 1
    `,
  )
    .bind(ticker)
    .first();
}

async function createJarRecord(ticker, env) {
  const insertResult = await env.DB.prepare(
    `
      INSERT INTO jars (ticker, display_name, note, comment, total_beans)
      VALUES (?, ?, ?, ?, 0)
    `,
  )
    .bind(
      ticker,
      ticker,
      "사용자 제안 병",
      "아직 마카의 코멘트가 준비되지 않았어요. 반응을 보면서 정리할 예정입니다.",
    )
    .run();

  return env.DB.prepare(
    `
      SELECT
        id,
        ticker,
        display_name,
        note,
        comment,
        total_beans,
        0 AS my_contribution
      FROM jars
      WHERE id = ?
      LIMIT 1
    `,
  )
    .bind(insertResult.meta?.last_row_id)
    .first();
}

let cachedApnsJwt = null;

async function sendApnsNotification({ token, environment, title, body }, env) {
  try {
    const jwt = await createApnsJwt(env);
    const host = environment === "production"
      ? "https://api.push.apple.com"
      : "https://api.sandbox.push.apple.com";

    const payload = {
      aps: {
        alert: {
          title,
          body,
        },
        sound: "default",
      },
    };

    const response = await fetch(`${host}/3/device/${token}`, {
      method: "POST",
      headers: {
        authorization: `bearer ${jwt}`,
        "apns-topic": env.APNS_BUNDLE_ID,
        "apns-push-type": "alert",
        "apns-priority": "10",
      },
      body: JSON.stringify(payload),
    });

    if (response.ok) {
      return { ok: true, status: response.status };
    }

    return {
      ok: false,
      status: response.status,
      error: await response.text(),
    };
  } catch (error) {
    return {
      ok: false,
      status: 500,
      error: error instanceof Error ? error.message : "Unknown APNs error",
    };
  }
}

async function createApnsJwt(env) {
  if (cachedApnsJwt && cachedApnsJwt.expiresAt > Date.now() + 60 * 1000) {
    return cachedApnsJwt.token;
  }

  const header = {
    alg: "ES256",
    kid: env.APNS_KEY_ID,
  };

  const payload = {
    iss: env.APNS_TEAM_ID,
    iat: Math.floor(Date.now() / 1000),
  };

  const signingInput = `${base64UrlEncodeJson(header)}.${base64UrlEncodeJson(payload)}`;
  const key = await importApnsPrivateKey(env.APNS_PRIVATE_KEY);
  const signature = await crypto.subtle.sign(
    {
      name: "ECDSA",
      hash: "SHA-256",
    },
    key,
    new TextEncoder().encode(signingInput),
  );

  const token = `${signingInput}.${base64UrlEncode(new Uint8Array(signature))}`;
  cachedApnsJwt = {
    token,
    expiresAt: Date.now() + 50 * 60 * 1000,
  };

  return token;
}

async function importApnsPrivateKey(pem) {
  const normalized = String(pem ?? "")
    .replace("-----BEGIN PRIVATE KEY-----", "")
    .replace("-----END PRIVATE KEY-----", "")
    .replace(/\s+/g, "");

  const binary = Uint8Array.from(atob(normalized), (char) => char.charCodeAt(0));

  return crypto.subtle.importKey(
    "pkcs8",
    binary.buffer,
    {
      name: "ECDSA",
      namedCurve: "P-256",
    },
    false,
    ["sign"],
  );
}

async function readJsonAsset(path, baseUrl, env) {
  const response = await env.ASSETS.fetch(new Request(new URL(path, baseUrl)));

  if (!response.ok) {
    throw new Error(`Failed to load asset: ${path}`);
  }

  return response.json();
}

function extractLatestPushTitle(meta) {
  const title =
    nestedString(meta, ["analysis_json", "core_thesis"]) ||
    nestedString(meta, ["analysis_json", "one_liner"]) ||
    "QQQ Maka 새 글이 올라왔어요";

  return truncateText(title, 60);
}

function extractLatestPushBody(meta) {
  const body =
    nestedString(meta, ["analysis_json", "one_liner"]) ||
    nestedString(meta, ["analysis_json", "post_close_recap"]) ||
    "앱에서 오늘의 새 글을 확인해 보세요.";

  return truncateText(body, 120);
}

function nestedString(source, path) {
  let current = source;

  for (const key of path) {
    if (!current || typeof current !== "object" || !(key in current)) {
      return "";
    }
    current = current[key];
  }

  return typeof current === "string" ? current.trim() : "";
}

function truncateText(text, maxLength) {
  const normalized = String(text ?? "").trim().replace(/\s+/g, " ");

  if (normalized.length <= maxLength) {
    return normalized;
  }

  return `${normalized.slice(0, maxLength - 1).trimEnd()}…`;
}

function base64UrlEncodeJson(value) {
  return base64UrlEncode(new TextEncoder().encode(JSON.stringify(value)));
}

function base64UrlEncode(bytes) {
  const binary = Array.from(bytes, (byte) => String.fromCharCode(byte)).join("");

  return btoa(binary)
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/g, "");
}

function extractBearerToken(authorizationHeader) {
  if (!authorizationHeader) {
    return null;
  }

  const match = authorizationHeader.match(/^Bearer\s+(.+)$/i);
  return match ? match[1].trim() : null;
}

function normalizeEmail(value) {
  return String(value ?? "")
    .trim()
    .toLowerCase();
}

function normalizeJarTicker(value) {
  const trimmed = String(value ?? "").trim();

  if (!trimmed) {
    return "";
  }

  if (trimmed.toLowerCase() === "maka") {
    return "Maka";
  }

  return trimmed.toUpperCase();
}

function isValidAdminRequest(request, env) {
  const adminCode = request.headers.get("X-Admin-Code");
  const expectedCode = env.ADMIN_PASSCODE ?? "12345";

  return Boolean(adminCode) && adminCode === expectedCode;
}

function isValidEmail(value) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
}

function parsePositiveInteger(value) {
  const parsed = Number.parseInt(String(value ?? ""), 10);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

function generateSixDigitCode() {
  return String(Math.floor(100000 + Math.random() * 900000));
}

function generateSessionToken() {
  const bytes = crypto.getRandomValues(new Uint8Array(32));
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function sha256Hex(value) {
  const encoded = new TextEncoder().encode(value);
  const digest = await crypto.subtle.digest("SHA-256", encoded);
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
}

function displayNameFromEmail(email) {
  const localPart = email.split("@")[0] || "maka";
  const cleaned = localPart
    .replace(/[._-]+/g, " ")
    .trim()
    .slice(0, 24);

  if (!cleaned) {
    return "Maka User";
  }

  return cleaned
    .split(" ")
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function buildAuthEmailText(code, ttlMinutes) {
  return [
    "Maka Workshop 인증코드입니다.",
    "",
    `인증코드: ${code}`,
    `유효 시간: ${ttlMinutes}분`,
    "",
    "이 코드를 앱에 입력하면 마카의 커피공방에 들어갈 수 있어요.",
  ].join("\n");
}

function buildAuthEmailHtml(code, ttlMinutes) {
  return `
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:24px;color:#1f1b18;">
      <h2 style="margin:0 0 16px;">Maka Workshop 인증코드</h2>
      <p style="margin:0 0 20px;">아래 코드를 앱에 입력하면 마카의 커피공방에 들어갈 수 있어요.</p>
      <div style="display:inline-block;padding:14px 18px;border-radius:14px;background:#f3ece4;font-size:28px;font-weight:700;letter-spacing:4px;">
        ${code}
      </div>
      <p style="margin:20px 0 0;color:#6b625c;">유효 시간은 ${ttlMinutes}분입니다.</p>
    </div>
  `;
}

function jsonResponse(data, init = {}) {
  return new Response(JSON.stringify(data, null, 2), {
    status: init.status ?? 200,
    headers: {
      "Content-Type": "application/json; charset=UTF-8",
      ...CORS_HEADERS,
      ...(init.headers ?? {}),
    },
  });
}
