// VeriSearch CI/CD. Owner: Ayesha Siddika, sole owner (PROJECT.md section 10).
// Dhrubo does not modify this file.
//
// Stage table is PROJECT.md section 10. Blocking stages run on every PR and
// every push to main. Nightly stages run only on a timer build of main.

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
        timeout(time: 60, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '30'))
        disableConcurrentBuilds(abortPrevious: true)
    }

    triggers {
        cron(env.BRANCH_NAME == 'main' ? 'H 2 * * *' : '')
    }

    environment {
        // uv resolves and installs into a project-local .venv.
        UV_PROJECT_ENVIRONMENT = '.venv'
        PYTHONUNBUFFERED = '1'
        // Unit and integration stages are CPU-only. Never install [train] here.
        PIP_DISABLE_PIP_VERSION_CHECK = '1'
    }

    stages {

        stage('setup') {
            steps {
                runCmd 'uv --version'
                runCmd 'uv venv --python 3.11'
                runCmd 'uv pip install -e ".[dev]"'
            }
        }

        stage('lint') {
            steps {
                runCmd 'uv run ruff check .'
                runCmd 'uv run ruff format --check .'
            }
        }

        stage('typecheck') {
            steps {
                runCmd 'uv run mypy --strict verisearch/'
            }
        }

        stage('unit') {
            steps {
                runCmd 'uv run pytest tests/ -m "not gpu and not integration" ' +
                       '--cov=verisearch --cov-report=xml --cov-report=term'
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'reports/junit*.xml'
                }
            }
        }

        stage('integration') {
            steps {
                runCmd 'uv run pytest tests/ -m integration'
            }
        }

        // ---- nightly, main only -------------------------------------------
        // Guarded on file existence so main stays green before P3 lands.

        stage('smoke-train') {
            when {
                allOf {
                    branch 'main'
                    expression { isNightly() }
                    expression { fileExists('scripts/02_train.sh') }
                }
            }
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
            steps {
                runCmd 'bash scripts/03_eval.sh --regression --n 100 --max-f1-drop 2.0'
            }
        }

        stage('artifact') {
            when { branch 'main' }
            steps {
                archiveArtifacts artifacts: 'coverage.xml', allowEmptyArchive: true
                archiveArtifacts artifacts: 'results/tables/**', allowEmptyArchive: true
                archiveArtifacts artifacts: 'results/figures/**', allowEmptyArchive: true
            }
        }
    }

    post {
        failure {
            echo 'Pipeline red. A red PR is not reviewable. A red main is top priority.'
        }
        cleanup {
            cleanWs(deleteDirs: true, notFailBuild: true)
        }
    }
}
