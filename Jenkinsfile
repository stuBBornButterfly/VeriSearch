// VeriSearch CI/CD. Owner: Ayesha Siddika, sole owner (PROJECT.md section 10).
// Dhrubo does not modify this file.
//
// Stage table is PROJECT.md section 10. Blocking stages run on every PR and
// every push to main. Nightly stages run only on a timer build of main.
//
// Plugin dependencies, deliberately minimal: Timestamper, AnsiColor, JUnit,
// Workspace Cleanup. Nothing here needs a plugin beyond those, because the
// controller is a laptop and every added plugin is another thing that fails to
// upgrade at the wrong moment.

def runCmd(String cmd) {
    if (isUnix()) {
        sh cmd
    } else {
        bat cmd
    }
}

def isNightly() {
    return currentBuild.getBuildCauses('hudson.triggers.TimerTrigger$TimerTriggerCause').size() > 0
}

pipeline {
    agent any

    options {
        timestamps()
        ansiColor('xterm')
        // Outer bound. Each stage has its own, tighter, timeout so that one
        // hung step cannot consume the whole budget and leave the rest of the
        // pipeline unreported.
        timeout(time: 45, unit: 'MINUTES')
        // artifactNumToKeepStr matters more than the log count: coverage and
        // eval artifacts accumulate, and this controller has ~80 GB free.
        buildDiscarder(logRotator(numToKeepStr: '30', artifactNumToKeepStr: '10'))
        disableConcurrentBuilds(abortPrevious: true)
    }

    triggers {
        cron(env.BRANCH_NAME == 'main' ? 'H 2 * * *' : '')
    }

    environment {
        UV_PROJECT_ENVIRONMENT = '.venv'
        PYTHONUNBUFFERED = '1'
        PIP_DISABLE_PIP_VERSION_CHECK = '1'
        // uv's default cache already lives outside the workspace, so cleanWs
        // does not touch it and each build reuses the downloads. Not overridden
        // here on purpose: a GString over an unset JENKINS_HOME would resolve
        // to a workspace-relative path that cleanWs then deletes, which would
        // look like it was working.
        //
        // Hardlinking the cache into the venv fails when they are on different
        // filesystems, which is the normal case here (cache on C:, workspace
        // wherever Jenkins put it). Copying is slower and always works.
        UV_LINK_MODE = 'copy'
        // PYTHONHASHSEED is deliberately NOT pinned. Leaving it random means a
        // test that accidentally depends on dict or set iteration order, or on
        // builtin hash(), will flake here and get found. Pinning it would make
        // CI agree with itself and disagree with reality.
        // verisearch/data/splits.py uses hashlib for exactly this reason.
    }

    stages {

        stage('setup') {
            options { timeout(time: 10, unit: 'MINUTES') }
            steps {
                runCmd 'uv --version'
                // `uv sync --locked` installs exactly uv.lock and FAILS if the
                // lock has drifted from pyproject.toml. That failure is the
                // point: it catches a dependency added without relocking,
                // which would otherwise make CI and both laptops disagree
                // about what is installed.
                //
                // --extra dev and nothing else. [retrieval] and [train] pull
                // torch and vLLM; the blocking stages are CPU-only
                // (section 10) and must stay installable in under a minute.
                //
                // --python 3.11 pins the interpreter. Without it uv honours
                // requires-python = ">=3.11" and may pick a newer one, so CI
                // would test a version that mypy's python_version = "3.11"
                // is not checking against.
                retry(2) {
                    runCmd 'uv sync --locked --extra dev --python 3.11'
                }
                // Prove the interpreter is what we asked for rather than
                // trusting the flag.
                runCmd 'uv run python -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version; print(sys.version)"'
            }
        }

        stage('lint') {
            options { timeout(time: 5, unit: 'MINUTES') }
            steps {
                runCmd 'uv run ruff check .'
                runCmd 'uv run ruff format --check .'
            }
        }

        stage('typecheck') {
            options { timeout(time: 10, unit: 'MINUTES') }
            steps {
                // No path argument: pyproject sets files = ["verisearch"] plus
                // disallow_any_explicit, which a path on the command line would
                // not change but which is easy to assume it does. One source of
                // truth, in the file both owners review.
                runCmd 'uv run mypy'
            }
        }

        stage('unit') {
            options { timeout(time: 15, unit: 'MINUTES') }
            steps {
                // Coverage gate (fail_under = 80) lives in pyproject, not here.
                runCmd 'uv run pytest tests/ -m "not gpu and not integration" ' +
                       '--junitxml=reports/junit-unit.xml ' +
                       '--cov=verisearch --cov-report=xml --cov-report=term'
            }
            post {
                always {
                    // allowEmptyResults: false on purpose. The previous version
                    // allowed empty and pytest was never asked for a junit
                    // file, so this step silently reported nothing on every
                    // build. An empty result now means the stage did not run
                    // the tests it claims to.
                    junit allowEmptyResults: false, testResults: 'reports/junit-unit.xml'
                }
            }
        }

        stage('integration') {
            options { timeout(time: 5, unit: 'MINUTES') }
            steps {
                // "and not gpu" is load-bearing: a test marked both
                // integration and gpu would otherwise run on this CPU-only
                // controller and fail for the wrong reason.
                //
                // pytest exits 5 when a marker matches nothing. That is a red
                // build on purpose - section 10 makes this stage blocking, so
                // "there are no integration tests" is a failure, not a pass.
                runCmd 'uv run pytest tests/ -m "integration and not gpu" ' +
                       '--junitxml=reports/junit-integration.xml'
            }
            post {
                always {
                    junit allowEmptyResults: false, testResults: 'reports/junit-integration.xml'
                }
            }
        }

        // ---- nightly, main only -------------------------------------------
        // The fileExists guards keep main green before P3 lands. They are
        // temporary and they fail quietly by design, which is the danger: once
        // scripts/02_train.sh and scripts/03_eval.sh exist, DELETE the
        // fileExists lines. Until then a typo in a script path looks identical
        // to "not written yet". Checked at P3 exit.

        stage('smoke-train') {
            when {
                allOf {
                    branch 'main'
                    expression { isNightly() }
                    expression { fileExists('scripts/02_train.sh') }
                }
            }
            options { timeout(time: 30, unit: 'MINUTES') }
            steps {
                runCmd 'bash scripts/02_train.sh --smoke --steps 5'
            }
        }

        stage('eval-regression') {
            when {
                allOf {
                    branch 'main'
                    expression { isNightly() }
                    expression { fileExists('scripts/03_eval.sh') }
                }
            }
            options { timeout(time: 60, unit: 'MINUTES') }
            steps {
                runCmd 'bash scripts/03_eval.sh --regression --n 100 --max-f1-drop 2.0'
            }
        }

        stage('artifact') {
            when { branch 'main' }
            options { timeout(time: 5, unit: 'MINUTES') }
            steps {
                // coverage.xml is NOT allowed to be empty: the unit stage
                // always produces it, so an empty archive means that stage
                // did not do what it reported.
                archiveArtifacts artifacts: 'coverage.xml', allowEmptyArchive: false
                archiveArtifacts artifacts: 'reports/junit-*.xml', allowEmptyArchive: false
                // These two are legitimately empty until P4 produces results.
                archiveArtifacts artifacts: 'results/tables/**', allowEmptyArchive: true
                archiveArtifacts artifacts: 'results/figures/**', allowEmptyArchive: true
            }
        }
    }

    post {
        failure {
            echo "FAILED ${env.JOB_NAME} #${env.BUILD_NUMBER} on ${env.BRANCH_NAME}"
            echo "commit ${env.GIT_COMMIT ?: 'unknown'}"
            echo 'A red PR is not reviewable (PROJECT.md section 10).'
            echo 'A red main is the top priority for both owners.'
            echo 'Workspace kept for this build - inspect it before rerunning.'
        }
        success {
            // Clean only on success. A failed build keeps its workspace so the
            // .venv, the coverage file and any partial output are still there
            // to look at - which is the whole reason you want a CI workspace.
            cleanWs(deleteDirs: true, notFailBuild: true)
        }
    }
}
